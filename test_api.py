"""
Self-contained API test for Hunyuan3D-2.

Starts the server automatically, runs image-to-3D and text-to-3D tests,
then shuts the server down.

Usage:
  uv run python test_api.py
  uv run python test_api.py --image "C:/path/to/image.jpg"
  uv run python test_api.py --text "a wooden chair"
  uv run python test_api.py --skip-text
  uv run python test_api.py --skip-image
  uv run python test_api.py --no-auto-server   # if server is already running
"""

import argparse
import base64
import json
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error

PORT = 8081
SERVER = f"http://localhost:{PORT}"
DEFAULT_IMAGE = r"C:\Users\kuncf\OneDrive\Pictures\DSC_5282 - Copy.jpg"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Server lifecycle ────────────────────────────────────────────────

def is_server_running() -> bool:
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def start_server(enable_t2i: bool = True) -> subprocess.Popen:
    """Start api_server.py as a subprocess and wait until it's healthy."""
    # Use 'uv run' so the server runs in Hunyuan3D-2's own managed venv,
    # not whatever Python is currently active.
    cmd = ["uv", "run", "python", "api_server.py", "--port", str(PORT)]
    if enable_t2i:
        cmd.append("--enable_t2i")

    print(f"Starting Hunyuan3D-2 server: {' '.join(cmd)}")
    print(f"  Working dir: {SCRIPT_DIR}")

    # Start the server process, inherit stderr so we see loading progress
    proc = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Wait for server to become healthy (model loading can take 30-120s)
    max_wait = 300  # 5 minutes for model download + loading
    start = time.time()
    print("  Waiting for server to be ready (loading models)...")
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            raise RuntimeError(f"Server process exited with code {proc.returncode}")
        if is_server_running():
            elapsed = time.time() - start
            print(f"  Server is ready! (took {elapsed:.1f}s)")
            return proc
        time.sleep(2)

    proc.kill()
    raise TimeoutError(f"Server did not become healthy within {max_wait}s")


def stop_server(proc: subprocess.Popen):
    """Gracefully stop the server process."""
    print("\nShutting down server...")
    if proc.poll() is not None:
        print("  Already stopped.")
        return

    try:
        if sys.platform == "win32":
            # On Windows, send CTRL_BREAK to the process group
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()

        proc.wait(timeout=10)
        print("  Server stopped gracefully.")
    except subprocess.TimeoutExpired:
        print("  Forcing kill...")
        proc.kill()
        proc.wait()
        print("  Server killed.")


def send_generate(payload: dict) -> str:
    """POST /send and return the uid."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{SERVER}/send",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    uid = data.get("uid")
    if not uid:
        raise RuntimeError(f"No uid in response: {data}")
    return uid


def poll_status(uid: str, timeout: float = 300) -> bytes:
    """Poll GET /status/{uid} until completed, return GLB bytes."""
    start = time.time()
    poll_interval = 2.0
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Generation timed out after {timeout}s")

        req = urllib.request.Request(f"{SERVER}/status/{uid}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        status = data.get("status")
        if status == "completed":
            model_b64 = data.get("model_base64")
            if not model_b64:
                raise RuntimeError("Status completed but no model_base64")
            return base64.b64decode(model_b64)
        elif status == "failed":
            raise RuntimeError(f"Generation failed: {data.get('error', 'unknown')}")
        else:
            print(f"  Status: {status} ({elapsed:.0f}s elapsed)")

        time.sleep(poll_interval)


def validate_glb(glb_bytes: bytes):
    """Basic GLB format validation."""
    if len(glb_bytes) < 12:
        raise ValueError(f"GLB too small: {len(glb_bytes)} bytes")

    magic, version, length = struct.unpack_from("<III", glb_bytes, 0)

    if magic != 0x46546C67:  # 'glTF'
        raise ValueError(f"Not a GLB file (magic: 0x{magic:08X})")

    print(f"  GLB magic:   0x{magic:08X} (glTF)")
    print(f"  GLB version: {version}")
    print(f"  GLB size:    {length:,} bytes")

    if version != 2:
        raise ValueError(f"Unexpected GLB version: {version}")

    if length != len(glb_bytes):
        print(f"  Warning: header says {length} bytes but got {len(glb_bytes)} bytes")

    # Read JSON chunk
    if len(glb_bytes) >= 20:
        chunk_len, chunk_type = struct.unpack_from("<II", glb_bytes, 12)
        if chunk_type == 0x4E4F534A:  # 'JSON'
            json_str = glb_bytes[20 : 20 + chunk_len].decode("utf-8", errors="replace")
            gltf = json.loads(json_str)
            meshes = gltf.get("meshes", [])
            nodes = gltf.get("nodes", [])
            print(f"  Meshes:      {len(meshes)}")
            print(f"  Nodes:       {len(nodes)}")
            if meshes:
                total_prims = sum(len(m.get("primitives", [])) for m in meshes)
                print(f"  Primitives:  {total_prims}")

    print("  GLB validation: PASSED")


def test_image_to_3d(image_path: str):
    print(f"\n{'='*60}")
    print(f"TEST: Image-to-3D")
    print(f"  Image: {image_path}")
    print(f"{'='*60}")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    print(f"  Image size: {len(image_b64) * 3 // 4:,} bytes")

    payload = {
        "image": image_b64,
        "seed": 42,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
    }

    print("  Sending to /send ...")
    uid = send_generate(payload)
    print(f"  Task UID: {uid}")

    print("  Polling for completion ...")
    start = time.time()
    glb_bytes = poll_status(uid)
    elapsed = time.time() - start
    print(f"  Generation completed in {elapsed:.1f}s")

    print("  Validating GLB ...")
    validate_glb(glb_bytes)

    out_path = "test_output_image.glb"
    with open(out_path, "wb") as f:
        f.write(glb_bytes)
    print(f"  Saved to: {out_path}")
    print("  TEST PASSED")


def test_text_to_3d(text: str):
    print(f"\n{'='*60}")
    print(f"TEST: Text-to-3D")
    print(f"  Prompt: {text}")
    print(f"{'='*60}")

    payload = {
        "text": text,
        "seed": 42,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
    }

    print("  Sending to /send ...")
    uid = send_generate(payload)
    print(f"  Task UID: {uid}")

    print("  Polling for completion ...")
    start = time.time()
    glb_bytes = poll_status(uid)
    elapsed = time.time() - start
    print(f"  Generation completed in {elapsed:.1f}s")

    print("  Validating GLB ...")
    validate_glb(glb_bytes)

    out_path = "test_output_text.glb"
    with open(out_path, "wb") as f:
        f.write(glb_bytes)
    print(f"  Saved to: {out_path}")
    print("  TEST PASSED")


def main():
    parser = argparse.ArgumentParser(description="Self-contained Hunyuan3D-2 API test")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Path to test image")
    parser.add_argument("--text", type=str, default="a simple wooden chair", help="Text prompt for text-to-3D")
    parser.add_argument("--skip-image", action="store_true", help="Skip image-to-3D test")
    parser.add_argument("--skip-text", action="store_true", help="Skip text-to-3D test")
    parser.add_argument("--no-auto-server", action="store_true", help="Don't start server (assume already running)")
    args = parser.parse_args()

    server_proc = None

    try:
        # Start server if needed
        if args.no_auto_server:
            if not is_server_running():
                print("FATAL: --no-auto-server specified but server is not running.")
                sys.exit(1)
            print("Using existing server.")
        elif is_server_running():
            print("Server already running, using it.")
        else:
            enable_t2i = not args.skip_text
            server_proc = start_server(enable_t2i=enable_t2i)

        passed = 0
        failed = 0

        if not args.skip_image:
            try:
                test_image_to_3d(args.image)
                passed += 1
            except Exception as e:
                print(f"  TEST FAILED: {e}")
                failed += 1

        if not args.skip_text:
            try:
                test_text_to_3d(args.text)
                passed += 1
            except Exception as e:
                print(f"  TEST FAILED: {e}")
                failed += 1

        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*60}")

    finally:
        # Always shut down server if we started it
        if server_proc is not None:
            stop_server(server_proc)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
