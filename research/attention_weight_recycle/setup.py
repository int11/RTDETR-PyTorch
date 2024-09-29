# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='attention_weight_twice_matmul',
    ext_modules=[
        CppExtension(
            name='attention_weight_twice_matmul',
            sources=['attention_weight_twice_matmul.cpp'],
            extra_compile_args=['-std=c++14']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# python setup.py build_ext --inplace