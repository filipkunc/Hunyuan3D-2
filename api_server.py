# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 enable_tex=False,
                 enable_t2i=False):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        self.enable_tex = enable_tex
        self.enable_t2i = enable_t2i
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        self.pipeline.enable_flashvdm(mc_algo='mc')
        self.pipeline_t2i = None
        if enable_t2i:
            self.pipeline_t2i = HunyuanDiTPipeline(
                'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                device=device
            )
        self.pipeline_tex = None
        if enable_tex:
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)
            # Use CPU offloading for texture pipeline to save VRAM when idle
            self.pipeline_tex.enable_model_cpu_offload()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        import time

        if 'image' in params:
            _set_progress(uid, 'Loading input image')
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            if 'text' in params:
                text = params["text"]
                if self.pipeline_t2i is None:
                    raise ValueError(
                        "Text-to-3D requires the server to be started with --enable_t2i. "
                        "Restart the server with: python api_server.py --enable_t2i"
                    )
                _set_progress(uid, f'Generating image from text')

                def _t2i_step_callback(pipe, step_index, timestep, callback_kwargs):
                    _set_progress(uid, f'Text-to-image: step {step_index + 1}/25')
                    return callback_kwargs

                image = self.pipeline_t2i(text, callback_on_step_end=_t2i_step_callback)
                _set_progress(uid, 'Text-to-image done')
            else:
                raise ValueError("No input image or text provided")

        if not params.get('no_rembg', False):
            _set_progress(uid, 'Removing background')
            image = self.rembg(image)
        else:
            logger.info(f"[{uid}] Skipping background removal (no_rembg=True)")

        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            seed = params.get("seed", 1234)
            octree_resolution = params.get("octree_resolution", 128)
            num_inference_steps = params.get("num_inference_steps", 5)
            guidance_scale = params.get('guidance_scale', 5.0)

            pipeline_kwargs = dict(
                image=image,
                generator=torch.Generator(self.device).manual_seed(seed),
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                mc_algo='mc',
            )
            _set_progress(uid, f'Shape generation (octree={octree_resolution}, steps={num_inference_steps})')
            start_time = time.time()
            mesh = self.pipeline(**pipeline_kwargs)[0]
            elapsed = time.time() - start_time
            _set_progress(uid, f'Shape generation done in {elapsed:.1f}s')

        if mesh is None:
            raise ValueError(
                "Failed to generate a valid 3D shape from this input. "
                "Try a different image or prompt, or lower the octree resolution."
            )

        if params.get('texture', False):
            if self.pipeline_tex is None:
                raise ValueError(
                    "Texture generation requires the server to be started with --enable_tex. "
                    "Restart the server with: python api_server.py --enable_tex"
                )
            _set_progress(uid, 'Cleaning mesh for texture')
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            _set_progress(uid, 'Generating texture')
            mesh = self.pipeline_tex(mesh, image)
            _set_progress(uid, 'Texture generation done')

        _set_progress(uid, 'Exporting model')
        output_type = params.get('type', 'glb')
        with tempfile.NamedTemporaryFile(suffix=f'.{output_type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{output_type}')
            mesh.export(save_path)

        torch.cuda.empty_cache()
        logger.info(f"[{uid}] Saved to {save_path}")
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


task_errors = {}    # uid -> error message
task_threads = {}   # uid -> Thread object
task_progress = {}  # uid -> {stage: str, gpu_used_gb: float, gpu_total_gb: float}


def _gpu_mem_info():
    """Return (used_gb, total_gb) for the current CUDA device."""
    try:
        free, total = torch.cuda.mem_get_info(0)
        used = (total - free) / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        return round(used, 1), round(total_gb, 1)
    except Exception:
        return 0.0, 0.0


def _set_progress(uid, stage):
    """Update progress stage and log GPU memory."""
    used, total = _gpu_mem_info()
    task_progress[str(uid)] = {
        'stage': stage,
        'gpu_used_gb': used,
        'gpu_total_gb': total,
    }
    logger.info(f"[{uid}] {stage}  [GPU: {used:.1f}/{total:.1f} GB]")


def _run_generate(uid, params):
    try:
        _set_progress(uid, 'Starting generation')
        worker.generate(uid, params)
        task_progress.pop(str(uid), None)
        logger.info(f"[{uid}] Generation thread completed")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"[{uid}] Generation failed: {e}")
        task_errors[str(uid)] = str(e)
        task_progress.pop(str(uid), None)


@app.get("/health")
async def health():
    return JSONResponse({'status': 'ok'}, status_code=200)


@app.post("/send")
async def send(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    uid_str = str(uid)
    t = threading.Thread(target=_run_generate, args=(uid, params,), daemon=True)
    task_threads[uid_str] = t
    t.start()
    ret = {"uid": uid_str}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    # Check for explicit error first
    if uid in task_errors:
        error_msg = task_errors.pop(uid)
        task_threads.pop(uid, None)
        response = {'status': 'failed', 'error': error_msg}
        return JSONResponse(response, status_code=200)

    # Check if output file exists (try .glb first, then .obj)
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    if not os.path.exists(save_file_path):
        save_file_path = os.path.join(SAVE_DIR, f'{uid}.obj')
    if os.path.exists(save_file_path):
        task_threads.pop(uid, None)
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)

    # Check if the worker thread is still alive
    thread = task_threads.get(uid)
    if thread is not None and not thread.is_alive():
        # Thread died without producing output or error (e.g. CUDA crash / segfault)
        task_threads.pop(uid, None)
        response = {'status': 'failed', 'error': 'Generation crashed unexpectedly (possible GPU out-of-memory). Check server logs.'}
        return JSONResponse(response, status_code=200)

    # Include progress stage and GPU info if available
    progress = task_progress.get(uid, {})
    response = {
        'status': 'processing',
        'stage': progress.get('stage', ''),
        'gpu_used_gb': progress.get('gpu_used_gb', 0),
        'gpu_total_gb': progress.get('gpu_total_gb', 0),
    }
    return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    parser.add_argument('--enable_t2i', action='store_true',
                        help='Enable text-to-image pipeline for text-to-3D generation (~6-8 GB extra VRAM)')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    # Log GPU info at startup
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name}, {gpu_total:.1f} GB VRAM")
    else:
        logger.warning("No CUDA GPU detected — generation will likely fail")

    worker = ModelWorker(model_path=args.model_path, device=args.device, enable_tex=args.enable_tex,
                         tex_model_path=args.tex_model_path, enable_t2i=args.enable_t2i)

    # Log VRAM after model loading
    used, total = _gpu_mem_info()
    logger.info(f"Models loaded. GPU memory: {used:.1f}/{total:.1f} GB")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
