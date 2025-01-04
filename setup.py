from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths


ext_modules = [
    CppExtension(
        "monotonic_align._C",
        ["monotonic_align_cpp/monotonic_align_cpp.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp", "-flto", "-pthread", "-static-libstdc++"],
        extra_link_args=["-O3", "-march=native", "-fopenmp", "-flto", "-pthread", "-static-libstdc++"]
    )
]

setup(
    name="monotonic_align",
    version='0.0.63',
    packages=find_packages(),
    # author='Ivan Shivalov',
    # packages=find_packages(),
    # author_email='ivansivalov396@gmail.com',
    # description='Module with search monotonic alignment algorithm',
    cmdclass={"build_ext": BuildExtension}, ext_modules=ext_modules,
)
