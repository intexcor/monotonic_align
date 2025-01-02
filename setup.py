from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths


ext_modules = [
      CppExtension(
            "monotonic_align",
            ["monotonic_align_cpp.cpp"],
            include_dirs=include_paths(),
      )
]

setup(
      name='monotonic_align',
      version='0.0.17',
      packages=find_packages(),
      author='Ivan Shivalov',
      author_email='ivansivalov396@gmail.com',
      description='Module with search monotonic alignment algorithm',
      cmdclass={"build_ext": BuildExtension}, ext_modules=ext_modules,
)
