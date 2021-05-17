from setuptools import setup
from setuptools_rust import Binding, RustExtension

import os


setup(name='granne',
      version='0.5.2',
      rust_extensions=[RustExtension('granne',
                                     'py/Cargo.toml', binding=Binding.RustCPython)],
      zip_safe=False,
      setup_requires=['setuptools_scm']
)
