from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(name='granne',
      version='0.4.0',
      rust_extensions=[RustExtension('granne',
                                     'py/Cargo.toml', binding=Binding.RustCPython)],
      zip_safe=False,
      setup_requires=['setuptools_rust>=0.8']
)
