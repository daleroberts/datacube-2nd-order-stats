#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

cimport openmp

from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport isnan, sqrt, acos, fabs

ctypedef fused floating:
    cnp.float32_t
    cnp.float64_t
    
ctypedef cnp.float32_t float32_t
ctypedef cnp.float64_t float64_t


def emad_core(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef float64_t total, value
    cdef size_t j, t, row, col

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):

                    # euclidean distance
                    total = 0.
                    for j in range(p):
                        value = X[row, col, j, t] - gm[row, col, j]
                        if not isnan(value):
                            total = total + value*value

                    result[row, col, t] = sqrt(total)
            

def smad_core(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef float64_t numer, norma, normb, value
    cdef size_t j, t, row, col

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):
                    
                    numer = 0.
                    norma = 0.
                    normb = 0.
                    
                    for j in range(p):
                        value = X[row, col, j, t]*gm[row, col, j]
                        numer = numer + value
                        norma = norma + X[row, col, j, t]*X[row, col, j, t]
                        normb = normb + gm[row, col, j]*gm[row, col, j]

                    result[row, col, t] = 1. - numer/(sqrt(norma)*sqrt(normb))
                    

def bcmad_core(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef float64_t numer, denom
    cdef size_t j, t, row, col

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):
                    
                    numer = 0.
                    denom = 0.
                    
                    for j in range(p):
                        numer = numer + fabs(X[row, col, j, t] - gm[row, col, j])
                        denom = denom + fabs(X[row, col, j, t] + gm[row, col, j])

                    result[row, col, t] = numer / denom
 

def emad(floating [:, :, :, :] X, floating [:,:,:] gm, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    result = np.empty((m, q, n), dtype=dtype)
    
    emad_core(X, gm, result, num_threads=num_threads)
    
    return np.median(result, axis=2)


def smad(floating [:, :, :, :] X, floating [:,:,:] gm, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    result = np.empty((m, q, n), dtype=dtype)
    
    smad_core(X, gm, result, num_threads=num_threads)
    
    return np.nanmedian(result, axis=2)


def bcmad(floating [:, :, :, :] X, floating [:,:,:] gm, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    result = np.empty((m, q, n), dtype=dtype)
    
    bcmad_core(X, gm, result, num_threads=num_threads)
    
    return np.nanmedian(result, axis=2)


def geomedian_core(floating [:, :, :, :] X, floating [:, :, :] mX, floating [:] w, 
               size_t maxiters=10000, floating eps=1e-6, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef size_t i, j, k, l, row, col
    cdef size_t nzeros, iteration
    cdef size_t reseed = 0
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *W
    cdef floating *T
    cdef floating *y
    cdef floating *y1
    cdef floating *R

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
    with nogil, parallel(num_threads=number_of_threads):
        Dinv = <floating *> malloc(sizeof(floating) * n)
        y1 = <floating *> malloc(sizeof(floating) * p)
        y = <floating *> malloc(sizeof(floating) * p)
        D = <floating *> malloc(sizeof(floating) * n)
        W = <floating *> malloc(sizeof(floating) * n)
        T = <floating *> malloc(sizeof(floating) * p)
        R = <floating *> malloc(sizeof(floating) * p)

        for row in prange(m, schedule='dynamic'):

            reseed = 1

            for col in range(q):

                # zero everything just to be careful for now...

                for j in range(p):
                    y1[j] = 0.0
                    y[j] = 0.0
                    T[j] = 0.0
                    R[j] = 0.0

                for i in range(n):
                    Dinv[i] = 0.0
                    D[i] = 0.0
                    W[i] = 0.0

                dist = 0.0
                Dinvs = 0.0
                total = 0.0
                r = 0.0
                rinv = 0.0
                Di = 0.0
                d = 0.0
                value = 0.0

                nzeros = 0



                if reseed == 1:

                    for j in range(p):
                        # nanmean
                        total = 0.
                        k = 0
                        for i in range(n):
                            value = X[row, col, j, i]
                            if not isnan(value):
                                total = total + value
                                k = k + 1
                        y[j] = total / k
                    
                iteration = 0                
                while iteration < maxiters:

                    for i in range(n):
                        
                        # euclidean distance
                        total = 0.
                        for j in range(p):
                            value = X[row, col, j, i] - y[j]
                            total = total + value*value
                        Di = sqrt(total)
                        
                        D[i] = Di
                        if not isnan(Di) and fabs(Di) > 0.:
                            Dinv[i] = w[i] / Di
                        else:
                            Dinv[i] = nan

                    # nansum
                    Dinvs = 0.
                    for i in range(n):
                        if not isnan(Dinv[i]):
                            Dinvs = Dinvs + Dinv[i]

                    for i in range(n):
                        W[i] = Dinv[i] / Dinvs

                    for j in range(p):
                        total = 0.
                        for i in range(n):
                            tmp = W[i] * X[row, col, j, i]
                            if not isnan(tmp):
                                total = total + tmp
                        T[j] = total

                    nzeros = n
                    for i in range(n):
                        if isnan(D[i]) or fabs(D[i]) > 0.:
                            nzeros = nzeros - 1

                    if nzeros == 0:
                        for j in range(p):
                            y1[j] = T[j]
                    elif nzeros == n:
                        break
                    else:
                        for j in range(p):
                            R[j] = (T[j] - y[j]) * Dinvs
                        
                        r = 0.
                        for j in range(p):
                            r = r + R[j]*R[j]
                        r = sqrt(r)
                        
                        if r > 0.:
                            rinv = nzeros/r
                        else:
                            rinv = 0.
                            
                        for j in range(p):
                            y1[j] = max(0, 1-rinv)*T[j] + min(1, rinv)*y[j]

                    total = 0.
                    for j in range(p):
                        value = y[j] - y1[j]
                        total = total + value*value
                    dist = sqrt(total)

                    for j in range(p):
                        y[j] = y1[j]
                        
                    iteration = iteration + 1
                
                    if isnan(dist):
                        reseed = 1
                        break
                    else:
                        reseed = 0

                    if dist < eps:
                        break

                for j in range(p):
                    mX[row, col, j] = y1[j]
            
        free(Dinv)
        free(y1)
        free(D)
        free(W)
        free(T)
        free(R)

def geomedian(floating [:, :, :, :] X, weight=None, maxiters=1000, floating eps=1e-4, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
        
    if weight is None:
        w = np.ones((n,), dtype=dtype)
    else:
        w = np.array(weight, dtype=dtype)

    result = np.empty((m, q, p), dtype=dtype)
    
    geomedian_core(X, result, w, maxiters=maxiters, eps=eps, num_threads=num_threads)
    
    return result
