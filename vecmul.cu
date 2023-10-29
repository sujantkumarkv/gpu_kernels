#include <stdio.h>

// cuda kernel for multiplication
__global__ void vecmul(float *a, float *b, float *d_temp, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        d_temp[i] = a[i] * b[i];
    }
}

// kernel for futher addition (reduction algo)
__global__ void reduction(float *d_temp, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N/2)
    {
        d_temp[i] = d_temp[i] + d_temp[N/2 - i -1];
        __syncthreads();
    }
}

int main() {
    int N = 50000;
    float *A, *B, *C;       // host
    float *d_A, *d_B, *d_C; // device
    float *d_temp;          // device temp
    // Allocate host memory
    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(1 * sizeof(float));

    // Allocate device memory
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, 1 * sizeof(float));
    cudaMalloc((void **)&d_temp, N * sizeof(float));
    // Initialize host arrays and copy to device
    for (int i = 0; i < N; ++i)
    {
        A[i] = float(i);
        B[i] = float(i);
    }
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vecmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_temp, N); // <<<...>>> syntax is used to specify the number of blocks and threads per block.

    while N > 1 {
        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        reduction<<<blocksPerGrid, threadsPerBlock>>>(d_temp, N);
        N = N / 2;
    }
    // Copy result back to host
    cudaMemcpy(C, d_temp, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("C: %f", *C);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    free(d_temp);
    return 0;
}
