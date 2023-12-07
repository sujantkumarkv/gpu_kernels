#include <stdio.h>

__global__ void mean (float* a, float* sum, float* mean, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(sum, a[i]);
    }
    __syncthreads();
    *mean = *sum / N;
}

__global__ void variance (float* a, float* mean, float* var, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if (i < N) {
        atomicAdd(sum, powf((a[i] - mean), 2));
    }
    __syncthreads();
    *var = sum / (N-1);
}

__global__ void layernorm (float* a, float* mean, float* var, float* layernorm, int N) {
    // constants / learnable parameters
    float epsilon = 1e-8, gamma = 1.0f, beta = 0.0f;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        layernorm[i] = ((a[i] - *mean) / sqrtf(*var + epsilon)) * gamma + beta;
    }
}

int main() {
    int N = 6;
    float *h_A, *h_mean, *h_var, *h_layernorm; // cpu
    float *d_A, *d_sum, *d_mean, *d_var, *d_layernorm; // gpu

    // memory allocation
    // host
    h_A = (float *)malloc(N * sizeof(float));
    h_mean = (float *)malloc(1 * sizeof(float));
    h_var = (float *)malloc(1 * sizeof(float)); 
    h_layernorm = (float *)malloc(N * sizeof(float));
    //device
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_sum, 1 * sizeof(float));
    cudaMalloc((void **)&d_mean, 1 * sizeof(float));
    cudaMalloc((void **)&d_var, 1 * sizeof(float));
    cudaMalloc((void **)&d_layernorm, N * sizeof(float));

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

    // later, atomicAdd
    float h_sum = 0.0f;
    cudaMemcpy(d_sum, &h_sum, 1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    

    // calculating kernel runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start
    cudaEventRecord(start, 0);

    // launch kernels
    mean<<< 1, N >>>(d_A, d_sum, d_mean, N);

    variance<<< 1, N >>>(d_A, d_sum, d_mean, d_var, N);
    layernorm<<< 1, N >>>(d_A, d_mean, d_var, d_layernorm, N);

    // stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // copy result back
    cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);
    // print
    printf("Layernorm:\n");
    for (int i=1; i < N; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");
    // time taken
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("\nFor N: %d, Time taken: %f ms\n", N, elapsed_time);
    printf("\n");
    // Cleanup
    cudaFree(d_A); cudaFree(d_sum);
    free(h_A);
    return 0;
}