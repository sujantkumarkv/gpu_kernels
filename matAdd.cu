#include <stdio.h>

// Define the CUDA kernel
__global__ void matAdd(float *a, float *b, float *c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int P=2, Q=3;
    int N = Q;
    float *h_A, *h_B, *h_C;     // host
    float *d_A, *d_B, *d_C;     // device
    // Allocate host memory
    h_A = (float *)malloc(P * Q * sizeof(float));
    h_B = (float *)malloc(P * Q * sizeof(float));
    h_C = (float *)malloc(P * Q * sizeof(float));

    // Allocate device memory
    cudaMalloc((void **)&d_A, P * Q * sizeof(float));
    cudaMalloc((void **)&d_B, P * Q * sizeof(float));
    cudaMalloc((void **)&d_C, P * Q * sizeof(float));

    // initialize host matrices & copy to device
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            h_A[i * Q + j] = float(i);
            h_B[i * Q + j] = float(j);
            h_C[i * Q + j] = 0.0f; // initialize to 0
        }
    }
    cudaMemcpy(d_A, h_A, P * Q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, P * Q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, P * Q * sizeof(float), cudaMemcpyHostToDevice);

    // for every row, invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (int i=0; i < P; i++) {
            float *d_A_row = &d_A[i * Q];
            float *d_B_row = &d_B[i * Q];
            float *d_C_row = &d_C[i * Q];
            // invoking kernel
            matAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_row, d_B_row, d_C_row, Q);
    }
    cudaMemcpy(h_C, d_C, P * Q * sizeof(float), cudaMemcpyDeviceToHost);

    // print
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            printf("%f", h_A[i * Q + j]);
        }
        printf("\n");
    }
    printf("\n\n");
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            printf("%f", h_B[i * Q + j]);
        }
        printf("\n");
    }
    printf("\n\n");
    for (int i=0; i < P; i++) {
        for (int j=0; j<Q; j++) {
            printf("%f", h_C[i * Q + j]);
        }
        printf("\n");
    }
    

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}