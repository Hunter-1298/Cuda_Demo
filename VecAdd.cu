#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// Threads to be run in parallel
#define CHECK_CUDA_ERROR(call)                                                                         \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

// Kernel definition
__global__ void VecAdd(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

int main()
{
    // Vector size (number of elements)
    int N = 50;
    size_t size = N * sizeof(float);

    // Allocate host memory, can use vectors but cuda built on C so pointers to  be consistent.
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Check if memory allocation was successful
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // <<<Blocks, Threads>>>
    int numThreads = 16;
    VecAdd<<<1, numThreads>>>(d_A, d_B, d_C, N);

    // Must copy result back from device (GPU) to host (CPU) before accessing
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numThreads; i++)
    {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    // Free the memory on the cuda device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free the memory on the host device
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}