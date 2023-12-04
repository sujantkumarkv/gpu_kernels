#include <stdio.h>
// cuda kernel for multiplication
__global__ void vecmul (float *a, float *b, float *d_temp, int N) {
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
    // imp. that all threads in a block reach this point before we proceed to final addition (incase of odd N)
    __syncthreads(); // it makes all threads wait
    if (N % 2 !=0 && i==0) {
        d_temp[0] += d_temp[N-1];
    }
}

int vecmul_func (float *d_A_row, float *d_B_col, float *d_C_ele, int N) {
    float *d_temp;
    cudaMalloc((void **)&d_temp, N * sizeof(float));

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blockDim(16, 16); // threadsPerBlock: 256
    dim3 gridDim((P + blockDim.x - 1)/blockDim.x, (Q + blockDim.y - 1)/blockDim.y);
    // time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    vecmul<<<blocksPerGrid, threadsPerBlock>>>(d_A_row, d_B_col, d_temp, N);

    while (N > 1) {
        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        reduction<<<blocksPerGrid, threadsPerBlock>>>(d_temp, N);
        N = N / 2;
    }
    // log off
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("\nTime taken: %f ms\n", elapsed_time);
    /**
     * copying a single float value from d_temp[0] to the location pointed by d_C_ele. 
     * This effectively stores the result of vecmul & reduction in the (i, j)-th element of matrix d_C
     * thus, updating d_C's values with the pointer d_C_ele by default.
    */
    cudaMemcpy(d_C_ele, d_temp, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp);
    return 0;
}

int main() {
    int P=2, Q=3, R=2;
    float *h_A, *h_B, *h_C;     // host
    float *d_A, *d_B, *d_C;     // device
    // Allocate host memory
    h_A = (float *)malloc(P * Q * sizeof(float));
    h_B = (float *)malloc(Q * R * sizeof(float));
    h_C = (float *)malloc(P * R * sizeof(float));

    // Allocate device memory
    cudaMalloc((void **)&d_A, P * Q * sizeof(float));
    cudaMalloc((void **)&d_B, Q * R * sizeof(float));
    cudaMalloc((void **)&d_C, P * R * sizeof(float));
    // cudaMalloc((void **)&d_temp, P * R * sizeof(float));

    // Initialize host arrays and copy to device
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < Q; ++j) {
            h_A[i * Q + j] = float(1);
        }
    }
    for (int i = 0; i < Q; ++i) {
        for (int j = 0; j < R; ++j) {
            h_B[i * R + j] = float(1);
        }
    }
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < R; ++j) {
            h_C[i * R + j] = 0.0f; // Initialize h_C to 0 as it will hold the result
        }
    }
    cudaMemcpy(d_A, h_A, P * Q * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Q * R * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, P * R * sizeof(float), cudaMemcpyHostToDevice);

    // invoke kernel for matrix rows & col
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < R; ++j) {
            float *d_A_row = &d_A[i * Q];
            float *d_B_col = &d_B[j];
            float *d_C_ele = &d_C[i * R + j];
            /**
             * d_C_ele is not a vector, it's a pointer to a single float value in the device memory. 
             * calling vecmul_func, we pass &d_C[i * N + j] as d_C_ele, which is (i, j)-th element in matrix C. 
             * So, d_C_ele points to a specific location in the d_C.
            */
            vecmul_func(d_A_row, d_B_col, d_C_ele, Q);
        }
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, P * R * sizeof(float), cudaMemcpyDeviceToHost);

    // result
    printf("result h_C:\n");
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < R; ++j) {
            printf("%f ", h_C[i * R + j]);
        }
        printf("\n");
    }
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

