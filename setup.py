from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="hssinfo",
    ext_modules=CUDAExtension(sources=["paddle_hssinfo.cu"]),
    include_dirs=["HSSInfo"],
)
