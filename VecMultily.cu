#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define N 96
// Function to take matrix A and B and return A*B
__global__ void VecMultiply(float *A, float *B, float *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id
    // num_threads/block * num_blocks allocated
    int stride = blockDim.x * gridDim.x;              // total number of threads
    
    // Each thread processes elements in steps of stride
    for(int i = tid; i < N; i += stride) {
        C[i] = A[i] * B[i];
    }
}



int main(void){ // entry point right here
    // Specify the size of memory we need to allocate
    auto size = N * sizeof(float);

    // Allocate 1xsize vectors
    float *h_A = (float*)malloc(size); 
    float *h_B = (float*)malloc(size); 
    float *h_C = (float*)malloc(size); 

    // Now provde dummy values for these vectors
    for (int i = 0; i < N; i++){
        h_A[i] = i * 1;
        h_B[i] = i * 2;
    }

    // Now we need to allocate memory for the variables on the remote device
    float *d_A; float *d_B; float *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy over the data
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int numThreads = 4;
    // <<<Blocks, Threads>>>
    VecMultiply<<<1, numThreads>>>(d_A, d_B, d_C);

    // Must copy result back from device (GPU) to host (CPU) before accessing
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
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