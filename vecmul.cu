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
        d_temp[i] += d_temp[i+N/2];
    }
    __syncthreads();

    /**
     * operation d_temp[0] += d_temp[N-1]; is actually safe to be performed by any thread, 
     * not just the thread with i == 0. This is because d_temp[N-1] is not being updated 
     * by other thread at the same time & d_temp[0] isn't being read by other thread at that time.
    */
    if (N % 2 !=0 && i==0) {
        d_temp[0] += d_temp[N-1];
     }
    //  __syncthreads();
}

int main() {
    int N = 5;
    float *A, *B, *C;       // host
    float *d_A, *d_B;       // device
    float *d_temp;          // device temp
    // Allocate host memory
    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(1 * sizeof(float));

    // Allocate device memory
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_temp, N * sizeof(float));
    // Initialize host arrays and copy to device
    for (int i = 0; i < N; ++i)
    {
        A[i] = float(i);
        B[i] = float(i+69);
    }
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vecmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_temp, N);

    while (N > 1) {
        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        reduction<<<blocksPerGrid, threadsPerBlock>>>(d_temp, N);
        N = N / 2;
    }
    // Copy result back to host
    cudaMemcpy(C, d_temp, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("C: %f\n", *C); // 720

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_temp);
    free(A);
    free(B);
    free(C);
    return 0;
}

