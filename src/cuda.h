#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
// Merge0309: cpp style header file
cudaStream_t get_cuda_stream();

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
// Merge0309: multiple preference of cuda calculation
enum {
	CUDNN_PREF_DEFAULT,
	CUDNN_PREF_FASTEST,
	CUDNN_PREF_SMALLEST
};
#endif

#endif
#endif
