from setuptools import setup
from setuptools_rust import Binding, RustExtension

import os


setup(name='cnstrc-granne',
      version='0.5.2.1',  # single-thread building
      rust_extensions=[RustExtension('granne',
                                     'py/Cargo.toml', binding=Binding.RustCPython)],
      zip_safe=False,
      setup_requires=['setuptools_scm']
)
