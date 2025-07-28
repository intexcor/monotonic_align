from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension


ext_modules = [
    CppExtension(
        "monotonic_align._C",
        ["monotonic_align/monotonic_align_cpp.cpp"],
        py_limited_api=True,
        extra_link_args = [],
        extra_compile_args={"cxx": ["-O3", "-fdiagnostics-color=always", "-DPy_LIMITED_API=0x03090000"]}
    )
]

setup(
    name="monotonic_align",
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension}, ext_modules=ext_modules,
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)