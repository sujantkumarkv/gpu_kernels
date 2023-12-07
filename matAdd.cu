#include <stdio.h>

// Define the CUDA kernel
__global__ void matAdd(float *a, float *b, float *c, int P, int Q) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < P && col < Q) {
        c[row * Q + col] = a[row * Q + col] + b[row * Q + col];
    }
}

int main() {
    int P=200000, Q=300000; // high values give segmentation fault
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
    
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 blockDim(16, 16); // threadsPerBlock: 256
    dim3 gridDim((P + blockDim.x - 1)/blockDim.x, (Q + blockDim.y - 1)/blockDim.y);

    // calculating kernel runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    // for (int i=0; i < P; i++) {
    //         float *d_A_row = &d_A[i * Q];
    //         float *d_B_row = &d_B[i * Q];
    //         float *d_C_row = &d_C[i * Q];
    //         // invoking kernel
    //         matAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_row, d_B_row, d_C_row, Q);
    // } 
    
    // invoking kernel
    matAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, P, Q);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_C, d_C, P * Q * sizeof(float), cudaMemcpyDeviceToHost);

    // print
    // for (int i=0; i < P; i++) {
    //     for (int j=0; j<Q; j++) {
    //         printf("%f", h_A[i * Q + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");
    // for (int i=0; i < P; i++) {
    //     for (int j=0; j<Q; j++) {
    //         printf("%f", h_B[i * Q + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");
    // for (int i=0; i < P; i++) {
    //     for (int j=0; j<Q; j++) {
    //         printf("%f", h_C[i * Q + j]);
    //     }
    //     printf("\n");
    // }

    // time taken
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("\nTime taken: %f ms\n", elapsed_time);
    

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}