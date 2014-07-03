from distutils.core import setup
from distutils.extension import Extension
from numpy.distutils.system_info import get_info, NotFoundError
from Cython.Distutils import build_ext
import os
import numpy

#try:
#    extra_info = get_info('mkl', 2)
#except NotFoundError:
#    extra_info = get_info('blas', 2)

mklroot = os.environ['MKLROOT']
codelib = os.environ['HOME']+'/code/lib'
codeinc = os.environ['HOME']+'/code/include'
extra_info = {
    'include_dirs': [mklroot+'/include', codeinc],
    'libraries': ['cvhf', 'mkl_intel_lp64', 'mkl_core', 'mkl_sequential', 'mkl_avx', 'pthread'],
    'library_dirs': [mklroot+'/lib/intel64', codelib],
    'extra_compile_args': ['-fopenmp','-ffast-math','-march=native'],
    'extra_link_args': ['-fopenmp'],
}

extra_info['include_dirs'].append(numpy.get_include())

ext_modules = [Extension('dmet_misc',
                         ['dmet_misc_helper.c', \
                          'ao2mo_helper.c', \
                          'mix_vhf_env.c', \
                          'dmet_misc.pyx', ],
                         **extra_info
                        )]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    script_args = ['build_ext', '--inplace']
)

# build .so with    python setup.py build_ext -i

