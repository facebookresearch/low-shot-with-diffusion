// -*- C++ -*-
/* Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */


// This is a SWIG wrapper for CSGMatrix.h

%module graph_utils;


%include <std_string.i>

typedef int int32_t;
typedef unsigned long uint64_t;

/*******************************************************************
 * Copied verbatim to wrapper
 *******************************************************************/

%{

#include <omp.h>

#include "CSRMatrix.h"

%}


/*******************************************************************
 * Parse headers
 *******************************************************************/

%newobject read_CSRMatrix;
%newobject CSRMatrix::slice_columns;
%newobject CSRMatrix::slice_nonzeros;
%newobject CSRMatrix::transpose;
%newobject CSRMatrix::point_op;
%newobject CSRMatrix::eliminate_zeros;
%newobject mul_matmat;



%include "CSRMatrix.h"


void *memcpy(void *dest, const void *src, size_t n);



/*******************************************************************
 * A way of providing float* pointers and friends to the functions
 * that require it (Lua version)
 *******************************************************************/


long *dummy_long();

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>


PyObject *swig_ptr (PyObject *a)
{
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}

 long *dummy_long() {
     return NULL;
 }

const char * swig_ptr_typename (PyObject *a) {
    void *outptr;
    if(SWIG_IsOK(SWIG_ConvertPtr(a, &outptr, SWIGTYPE_p_float, 0 |  0 )))
        return "float32";
    if(SWIG_IsOK(SWIG_ConvertPtr(a, &outptr, SWIGTYPE_p_long, 0 |  0 )))
        return "int64";
    if(SWIG_IsOK(SWIG_ConvertPtr(a, &outptr, SWIGTYPE_p_unsigned_long, 0 |  0 )))
        return "int64";
    if(SWIG_IsOK(SWIG_ConvertPtr(a, &outptr, SWIGTYPE_p_int, 0 |  0 )))
        return "int32";
    return "unknown";
}

%}


%init %{
    /* needed, else crash at runtime */
    import_array();
%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

const char * swig_ptr_typename (PyObject *a);



// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(prefix, ctype, numpytype)

%{
    PyObject * prefix ## _rev_swig_ptr(void *src, npy_intp size) {
        return PyArray_SimpleNewFromData(1, &size, numpytype, src);
    }
%}

PyObject * prefix ## _rev_swig_ptr(void *src, size_t size);

%enddef

REV_SWIG_PTR(fvec, float, NPY_FLOAT32);
REV_SWIG_PTR(ivec, int, NPY_INT32);
REV_SWIG_PTR(bvec, unsigned char, NPY_UINT8);
REV_SWIG_PTR(ulvec, unsigned long, NPY_UINT64);
REV_SWIG_PTR(lvec, long, NPY_INT64);



/*******************************************************************
 * We want to see these types of pointers (does not use script
 * matrices, so mainly for debugging)
 *******************************************************************/

%include "carrays.i"

%array_class (float, FloatArray)
%array_class (double, DoubleArray)
%array_class (int, IntArray)
%array_class (uint64_t, Uint64Array)
