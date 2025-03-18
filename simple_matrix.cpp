#include <iostream>
#include <memory>
#include <cuda_runtime>

// Adds vectors of A and B, size on N into C
// ThreadIDX is a built in varaible
__global__ void VecAdd(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = B[i] + A[i];
}

// Thread IDX is a three component vector,
// Thread block dimensions Dx,Dy,Dz
// Thread idx = (x+ yDx +zDxDy)
__global__ void VecAddThreeComponent(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // VecAdd function
    // Sets number of threads in <<<...>>>
    // VecAdd<<<1, N>>>(A, B, C);

    // VecAdd 3 dimension
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    VecAddThreeComponent<<<numBlocks, threadsPerBlock>>>(A, B, C);

    return 0;
}