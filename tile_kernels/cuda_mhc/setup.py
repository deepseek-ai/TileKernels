import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='cuda_mhc',
    ext_modules=[
        CUDAExtension(
            name='_mhc_cuda',
            sources=[
                'python/bindings.cpp',
                'src/mhc_expand.cu',
                'src/mhc_pre_fused.cu',
                'src/mhc_post.cu',
                'src/mhc_head.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++20'],
                'nvcc': [
                    '-O3',
                    '-std=c++20',
                    '--ptxas-options=-v',
                    '-lineinfo',
                    '-arch=sm_90',
                ],
            },
            include_dirs=[
                '/usr/local/lib/python3.11/site-packages/nvidia/cu13/include',
                '/usr/local/lib/python3.11/site-packages/nvidia/cu13/include/cccl',
                os.path.abspath(os.path.join(os.path.dirname(__file__), 'include')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')),
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
