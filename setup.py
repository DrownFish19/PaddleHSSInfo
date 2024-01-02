from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="hssinfo",
    ext_modules=CUDAExtension(
        sources=["paddle_hssinfo.cu", "HSSInfo/hssinfo.cc", "HSSInfo/hssinfo.cu"]
    ),
)
