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

import os
import random

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


class HunyuanDiTPipeline:
    def __init__(
        self,
        model_path="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        device='cuda'
    ):
        self.device = device
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                enable_pag=True,
                pag_applied_layers=["blocks.(16|17|18|19)"],
                safety_checker=None,
            ).to(device)
            self.use_pag = True
        except (ValueError, TypeError):
            # PAG may not be compatible with this diffusers version
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(device)
            self.use_pag = False
        # Ensure safety checker is fully disabled (some pipelines re-attach it)
        if hasattr(self.pipe, 'safety_checker'):
            self.pipe.safety_checker = None
        # self.pos_txt = ",白色背景,3D风格,最佳质量"
        self.pos_txt = "3D"
        self.neg_txt = ""
        # self.neg_txt = "文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态," \
        #                "残缺,多余的手指,变异的手,画得不好的手,画得不好的脸,变异,畸形,模糊,脱水,糟糕的解剖学," \
        #                "糟糕的比例,多余的肢体,克隆的脸,毁容,恶心的比例,畸形的肢体,缺失的手臂,缺失的腿," \
        #                "额外的手臂,额外的腿,融合的手指,手指太多,长脖子"

    def compile(self):
        # accelarate hunyuan-dit transformer,first inference will cost long time
        torch.set_float32_matmul_precision('high')
        self.pipe.transformer = torch.compile(self.pipe.transformer, fullgraph=True)
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, fullgraph=True)
        generator = torch.Generator(device=self.pipe.device)  # infer once for hot-start
        extra_kwargs = {'pag_scale': 1.3} if self.use_pag else {}
        out_img = self.pipe(
            #prompt='美少女战士',
            prompt='',
            #negative_prompt='模糊',
            negative_prompt='',
            num_inference_steps=25,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False,
            **extra_kwargs
        )[0][0]

    @torch.no_grad()
    def __call__(self, prompt, seed=0, callback_on_step_end=None):
        seed_everything(seed)
        generator = torch.Generator(device=self.pipe.device)
        generator = generator.manual_seed(int(seed))
        extra_kwargs = {'pag_scale': 1.3} if self.use_pag else {}
        if callback_on_step_end is not None:
            extra_kwargs['callback_on_step_end'] = callback_on_step_end
        out_img = self.pipe(
            prompt=prompt[:60] + self.pos_txt,
            negative_prompt=self.neg_txt,
            num_inference_steps=25,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False,
            **extra_kwargs
        )[0][0]
        return out_img
