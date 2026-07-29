#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "/home/cv/anaconda3")
CUDA_LIBRARY_DIRS = [
    os.path.join(ROOT_DIR, "cuda_lib"),
    os.path.join(CONDA_PREFIX, "lib"),
    "/home/cv/anaconda3/lib",
]

setup(
    name="diff_gaussian_rasterization_feature",
    packages=['diff_gaussian_rasterization_feature'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_feature._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            library_dirs=CUDA_LIBRARY_DIRS,
            #extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            ### To fix illegal memory issue
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
