from distutils.core import setup, Extension
from distutils.command.build import build
import os

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# We need a custom build order in order to ensure that topology.py is available
# before we try to copy it to the target location
class CustomBuild(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

swig_opts=['-c++']
extra_compile_args=['-std=c++11']
setup(name='ngl',
      version='0.1',
      description='A library for computing neighborhood graphs',
      ext_modules=[Extension('_ngl',['ngl.i','GraphStructure.cpp', 'UnionFind.cpp'],
                             swig_opts=swig_opts,
                             extra_compile_args=extra_compile_args)],
      py_modules=['ngl'],
      cmdclass={'build': CustomBuild})
