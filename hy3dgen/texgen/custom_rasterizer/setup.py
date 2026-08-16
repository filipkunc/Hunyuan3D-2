import sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# build custom rasterizer
# build with `python setup.py install`
# nvcc is needed

if sys.platform == 'win32':
    # MSVC flags for the Windows build.
    extra_compile_args = {
        'cxx': ['/Zc:__cplusplus', '/permissive-'],
        'nvcc': [
            '--allow-unsupported-compiler',
            '-Xcompiler', '/Zc:__cplusplus',
            '-Xcompiler', '/permissive-',
        ],
    }
else:
    # GCC/Clang flags for Linux/macOS. The MSVC-only switches above are not
    # understood by g++ (it would treat '/Zc:__cplusplus' as a file path).
    extra_compile_args = {
        'cxx': ['-O3'],
        'nvcc': ['-O3', '--allow-unsupported-compiler'],
    }

custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
], extra_compile_args=extra_compile_args)

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
