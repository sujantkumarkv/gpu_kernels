#include <stdio.h>
#include <stdlib.h>

// ReLU kernel
__global__ void leakyrelu(float *a, float *b, float alpha, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// leaky ReLU
	if (i < N) {
		b[i] = fmaxf(alpha * a[i], a[i]);
	}
}

int main() {
	int N = 5;
	float *h_A, *h_B;
	float *d_A, *d_B;
	float alpha = 0.01;

	// host memory
	h_A = (float *)malloc(N * sizeof(float));
	h_B = (float *)malloc(N * sizeof(float));

	// device memory
	cudaMalloc(&d_A, N * sizeof(float));
	cudaMalloc(&d_B, N * sizeof(float));

	// initialize
	for (int i = 0; i < N; ++i) {
		h_A[i] = (float)(rand() % 10 - 4);
	}
	
	// copy host data to device
	cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
        
	// launch kernel instance
	dim3 blockDim(256);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x);
	relu<<<gridDim, blockDim>>>(d_A, d_B, alpha, N);
	
	// copy result back to host
	cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

  	// display results
	printf("Vector A: \n");
	for (int i = 0; i < N; ++i) {
		printf("%f \n", h_A[i]);
	}
	printf("\n");
	printf("ReLU: \n");
	for (int i = 0; i < N; ++i) {
		printf("%f \n", h_B[i]);
	}
    printf("\n");
	
	// clean up data
	free(h_A); free(h_B);
	cudaFree(d_A); cudaFree(d_B);

	return 0;
}