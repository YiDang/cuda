#include <stdio.h>  
#include <stdlib.h>  
#include <cuda_runtime.h>  
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#define M_ 2  
#define N_ 4 
#define P_ 3 

#define MAX 100
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  

#define show(matrix, lenm, lenn) for (r = 0; r < lenm; r++){for (c = 0; c < lenn; c++){printf("%.6f ", matrix[r*lenn+c]);}printf("\n");}printf("\n");

__global__ void InitMatrix(float **m, unsigned int rows, unsigned int cols, int seed)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
    curandState_t state;
    curand_init((row*cols + col) * seed, /* the seed controls the sequence of random values that are produced */
              row, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);
    if (row < rows && col < cols)  
    {  
        //m[row][col] = curand_uniform(&state);
        m[row][col] = 1;
    }  
}  

__global__ void Multiply(float **mA, float **mB, float **mC, unsigned int m, unsigned int n, unsigned int p)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
 	
 	
 	if (row < m && col < p)  
    { 
    	mC[row][col] = 0;
	    for(int i = 0; i < n; i++){
	    	mC[row][col] += mA[row][i] * mB[i][col];
	    }
    }
}

void cuda(float *host_array_A, float *host_array_B, float *host_array_C)
{
	float **device_matrix_A = NULL;  
    float **host_matrix_A = NULL;  
    float *device_array_A = NULL;  
    //float *host_array_A = NULL;  
    cudaError_t res;  
    int r, c;    
  
    res = cudaMalloc((void**)(&device_matrix_A), M_*sizeof(float*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_A), M_*N_*sizeof(float));CHECK(res)  
    host_matrix_A = (float**)malloc(M_*sizeof(float*));  
    //host_array_A = (float*)malloc(M_*N_*sizeof(float));  
  
    for (r = 0; r < M_; r++)  
    {  
        host_matrix_A[r] = device_array_A + r*N_;  
    }  

    res = cudaMemcpy((void*)(device_matrix_A), (void*)(host_matrix_A), M_*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)  
    dim3 dimBlock(16,16);  
    dim3 dimGrid((N_+dimBlock.x-1)/(dimBlock.x), (M_+dimBlock.y-1)/(dimBlock.y));  
    InitMatrix<<<dimGrid, dimBlock>>>(device_matrix_A, M_, N_, 1);  
    res = cudaMemcpy((void*)(host_array_A), (void*)(device_array_A), M_*N_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  

    float **device_matrix_B = NULL;  
    float **host_matrix_B = NULL;  
    float *device_array_B = NULL;  
    //float *host_array_B = NULL;  
  
    res = cudaMalloc((void**)(&device_matrix_B), N_*sizeof(float*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_B), N_*P_*sizeof(float));CHECK(res)  
    host_matrix_B = (float**)malloc(N_*sizeof(float*));  
    //host_array_B = (float*)malloc(N_*P_*sizeof(float));  
  
    for (r = 0; r < N_; r++)  
    {  
        host_matrix_B[r] = device_array_B + r*P_;  
    }  

    res = cudaMemcpy((void*)(device_matrix_B), (void*)(host_matrix_B), N_*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)   
    InitMatrix<<<dimGrid, dimBlock>>>(device_matrix_B, N_, P_, 2);  
    res = cudaMemcpy((void*)(host_array_B), (void*)(device_array_B), N_*P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  

    float **device_matrix_C = NULL;  
    float **host_matrix_C = NULL;  
    float *device_array_C = NULL;  
    //float *host_array_C = NULL;  

    res = cudaMalloc((void**)(&device_matrix_C), M_*sizeof(float*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_C), M_*P_*sizeof(float));CHECK(res)  
    host_matrix_C = (float**)malloc(M_*sizeof(float*));  
    //host_array_C = (float*)malloc(M_*P_*sizeof(float));  

    for (r = 0; r < M_; r++)  
    {  
        host_matrix_C[r] = device_array_C + r*P_;  
    } 

    res = cudaMemcpy((void*)(device_matrix_C), (void*)(host_matrix_C), M_*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res) 
    Multiply<<<dimGrid, dimBlock>>>(device_matrix_A, device_matrix_B, device_matrix_C, M_, N_, P_);  
    res = cudaMemcpy((void*)(host_array_C), (void*)(device_array_C), M_*P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  
  
    cudaFree((void*)device_matrix_A);  
    cudaFree((void*)device_array_A);  
    cudaFree((void*)device_matrix_B);  
    cudaFree((void*)device_array_B);  
    cudaFree((void*)device_matrix_C);  
    cudaFree((void*)device_array_C); 
    free(host_matrix_A);  
    //free(host_array_A);  
    free(host_matrix_B);  
   	//free(host_array_B); 
    free(host_matrix_C);  
    //free(host_array_C); 
}  

void initArray(float *array, int len)
{
	for(int i = 0; i < len; i++){

	}
}
void sequential(float *host_array_A, float *host_array_B, float *host_array_C)
{
	for(int i = 0; i < M_; i++)
	{
		for(int j = 0; j < P_; j++)
		{	
			host_array_C[i * M_ + j] = 0;
			for(int k = 0; k < N_; k++)
			{
				host_array_C[i * M_ + j] += host_array_A[i * M_ + k] * host_array_B[k * N_ + j];
			}
		}
	}
}

void cublas(float *host_array_A, float *host_array_B, float *host_array_C)
{
	int lda = M_, ldb = N_, ldc = M_;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

 	printf("start\n");

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M_, N_, P_, alpha, host_array_A, lda, host_array_B, ldb, beta, host_array_C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}
int main(int argc, char **argv)  
{  
	int r, c;
	float *host_array_A = (float*)malloc(M_*N_*sizeof(float)); 
	float *host_array_B = (float*)malloc(P_*N_*sizeof(float));
	float *host_array_C_para = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_seq = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_cublas = (float*)malloc(M_*P_*sizeof(float));
	cuda(host_array_A, host_array_B, host_array_C_para);

	show(host_array_A, M_, N_);
	show(host_array_B, N_, P_);
	show(host_array_C_para, M_, P_);

	sequential(host_array_A, host_array_B, host_array_C_seq);

	show(host_array_C_seq, M_, P_);

    cublas(host_array_A, host_array_B, host_array_C_cublas);

    show(host_array_C_cublas, M_, P_);
    
	free(host_array_A); 
	free(host_array_B); 
	free(host_array_C_para); 
	free(host_array_C_seq); 
	free(host_array_C_cublas); 
    return 0;  
}  