#include <stdio.h>

__global__ void exponentiate (float* a, int N) {
    // softmax kernel
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // exponentiated
    if (i < N) {
        a[i] = __expf(a[i]);
    }
}

__global__ void reduction (float* a, float* sum, int N) {
    // calculate thread ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // boundary check
    if (i < N) {
        atomicAdd(sum, a[i]);
    }
}

__global__ void softmax (float* a, float* sum, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // boundary check
    if (i < N) {
        a[i] /= *sum;
    }
}

int main() {
    int N = 6;
    float *h_A; // cpu
    float *d_A, *d_sum; // gpu

    // memory allocation
    // host
    h_A = (float *)malloc(N * sizeof(float));
    //device
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_sum, sizeof(float));

    float h_sum = 0.0f;
    // initialize host vectors & copy to device
    for (int i=1; i < N; i++) {
        h_A[i] = float(i);
    }
    // print initially
    printf("A:\n");
    for (int i=1; i < N; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");

    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    // calculating kernel runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // launch kernels
    exponentiate<<< 1, N >>>(d_A, N);
    reduction<<< 1, N >>>(d_A, d_sum, N);
    softmax<<< 1, N >>>(d_A, d_sum, N);

    // stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // copy result back
    cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);
    // print
    printf("Softmax(A):\n");
    for (int i=1; i < N; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");
    // time taken
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("\nTime taken: %f ms\n", elapsed_time);
    printf("\n");
    // Cleanup
    cudaFree(d_A); cudaFree(d_sum);
    free(h_A);
    return 0;
}