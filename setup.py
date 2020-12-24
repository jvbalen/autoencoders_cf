
import os
import sys
import glob
import platform

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


use_openmp = True


def define_extensions():
    """Adapted from https://github.com/benfred/implicit
    """
    if sys.platform.startswith("win"):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ["/O2", "/openmp"]
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = "/usr/local/opt/gcc/lib/gcc/" + gcc[-1] + "/"
            link_args = ["-Wl,-rpath," + rpath]
        else:
            link_args = []

        compile_args = ["-Wno-unused-function", "-Wno-maybe-uninitialized", "-O3", "-ffast-math"]
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

        compile_args.append("-std=c++11")
        link_args.append("-std=c++11")

    modules = [Extension("extensions", ["extensions.pyx"],
                         language="c++",
                         extra_compile_args=compile_args,
                         extra_link_args=link_args)]

    return cythonize(modules)


def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9]*.[0-9]*",
        "/opt/local/bin/g++-mp-[0-9]*",
        "/usr/local/bin/g++-[0-9]*.[0-9]*",
        "/usr/local/bin/g++-[0-9]*",
    ]
    if platform.system() == "Darwin":
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


setup(name='autoencoders_cf',
      version='1.0.0',
      url='https://github.com/jvbalen/autoencoders_cf',
      author='Jan Van Balen',
      author_email='jvanbalen@uantwerpen.be',
      description='Autoencoders for Collaborative Filtering',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn',
                        'tensorflow>=1.15.2', 'gin-config', 'tqdm'],
      extras_require={'test': ['pytest']},
      setup_requires=["Cython>=0.24"],
      ext_modules=cythonize("extensions.pyx"))
