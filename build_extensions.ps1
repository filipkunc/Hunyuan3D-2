# Build Hunyuan3D-2 native extensions (run from Hunyuan3D-2/ directory)
# Prerequisites:
#   - CUDA Toolkit 12.8+ installed (https://developer.nvidia.com/cuda-12-8-0-download-archive)
#   - Visual Studio Build Tools with C++ workload
#   - Python venv already created (uv sync)

$ErrorActionPreference = "Stop"

$VENV_PYTHON = "$PSScriptRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Error "Python venv not found at $VENV_PYTHON — run 'uv sync' first"
    exit 1
}

# 1. Build differentiable_renderer (CPU, pybind11 — no CUDA needed)
Write-Host "`n=== Building differentiable_renderer ===" -ForegroundColor Cyan
Push-Location "$PSScriptRoot\hy3dgen\texgen\differentiable_renderer"
& $VENV_PYTHON setup.py build_ext --inplace
if ($LASTEXITCODE -ne 0) { Write-Error "differentiable_renderer build failed"; Pop-Location; exit 1 }
Pop-Location
Write-Host "differentiable_renderer built successfully" -ForegroundColor Green

# 2. Build custom_rasterizer (CUDA — requires nvcc)
Write-Host "`n=== Building custom_rasterizer ===" -ForegroundColor Cyan

# Auto-detect CUDA_HOME if not set
if (-not $env:CUDA_HOME) {
    $cudaPaths = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    if ($cudaPaths) {
        $env:CUDA_HOME = $cudaPaths[0].FullName
        Write-Host "Auto-detected CUDA_HOME: $env:CUDA_HOME" -ForegroundColor Yellow
    } else {
        Write-Error @"
CUDA Toolkit not found. Install CUDA Toolkit 12.8 from:
  https://developer.nvidia.com/cuda-12-8-0-download-archive
  (Custom install: only need Compiler + Libraries, skip driver update)
Then set CUDA_HOME or re-run this script.
"@
        exit 1
    }
}

$nvcc = Join-Path $env:CUDA_HOME "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
    Write-Error "nvcc.exe not found at $nvcc — CUDA Toolkit may not be properly installed"
    exit 1
}
Write-Host "Using nvcc: $nvcc"

Push-Location "$PSScriptRoot\hy3dgen\texgen\custom_rasterizer"
& $VENV_PYTHON setup.py install
if ($LASTEXITCODE -ne 0) { Write-Error "custom_rasterizer build failed"; Pop-Location; exit 1 }
Pop-Location
Write-Host "custom_rasterizer built successfully" -ForegroundColor Green

Write-Host "`n=== All extensions built successfully ===" -ForegroundColor Green
Write-Host "You can now start the server with: Start Hunyuan3D Server (with Textures)"
