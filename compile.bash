# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -ex

swig -c++ -python graph_utils.swig


MKL_ROOT=/public/apps/intel/mkl/2018.0.128/mkl/


g++ -c -fPIC -g -Wall -O3 -std=c++11 -fopenmp \
    -I $MKL_ROOT/include -Wno-sign-compare CSRMatrix.cpp

g++ -c -fPIC -g -std=c++11 \
    -I $( python -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()" ) \
    -I $( python -c "import numpy ; print numpy.get_include()" ) \
    graph_utils_wrap.cxx


g++ -shared -o _graph_utils.so graph_utils_wrap.o CSRMatrix.o -fopenmp \
    -L $MKL_ROOT/lib/intel64 -lmkl_gf_lp64 \
    -lmkl_core -lmkl_gnu_thread -lmkl_avx2 -ldl -lpthread
