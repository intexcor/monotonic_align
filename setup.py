from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension


ext_modules = [
      CppExtension(
            "monotonic_align",
            ["monotonic_align_cpp.cpp"],

      )
]

setup(
      name='monotonic_align',
      version='0.0.25',
      author='Ivan Shivalov',
      packages=find_packages(),
      author_email='ivansivalov396@gmail.com',
      description='Module with search monotonic alignment algorithm',
      cmdclass={"build_ext": BuildExtension}, ext_modules=ext_modules,
      extra_objects=["monotonic_align.pyi"]
)