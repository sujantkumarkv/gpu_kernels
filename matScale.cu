#include <stdio.h>

// Define the CUDA kernel
__global__ void matScale(float *a, float *b, float scale, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        b[i] = a[i] / scale;
    }
}

int main() {
    int P=2, Q=3;
    int N = Q;
    float scale = 10.0f;
    float *h_A, *h_B;     // host
    float *d_A, *d_B;     // device
    // Allocate host memory
    h_A = (float *)malloc(P * Q * sizeof(float));
    h_B = (float *)malloc(P * Q * sizeof(float));

    // Allocate device memory
    cudaMalloc((void **)&d_A, P * Q * sizeof(float));
    cudaMalloc((void **)&d_B, P * Q * sizeof(float));

    // initialize host matrices & copy to device
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            h_A[i * Q + j] = float(i);
            h_B[i * Q + j] = float(j);
        }
    }
    cudaMemcpy(d_A, h_A, P * Q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, P * Q * sizeof(float), cudaMemcpyHostToDevice);

    // for every row, invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (int i=0; i < P; i++) {
            float *d_A_row = &d_A[i * Q];
            float *d_B_row = &d_B[i * Q];
            // invoking kernel
            matScale<<<blocksPerGrid, threadsPerBlock>>>(d_A_row, d_B_row, scale, Q);
    }
    cudaMemcpy(h_B, d_B, P * Q * sizeof(float), cudaMemcpyDeviceToHost);

    // print
    printf("scale: %f \n", scale);
    printf("Matrix A \n");
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            printf("%f ", h_A[i * Q + j]);
        }
        printf("\n");
    }
    printf("\n Matrix B (scaled A)\n");
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            printf("%f ", h_B[i * Q + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);

    return 0;
}