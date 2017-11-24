#include <stdio.h>  
#include <stdlib.h>  
#include <cuda_runtime.h>  
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

#define M_ 2  
#define N_ 4 
#define P_ 3 

#define MAX 100
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  
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
        m[row][col] = curand_uniform(&state);
        //m[row][col] = 1;
    }  
}  

__global__ void Multiply(float **mA, float **mB, float **mC, unsigned int m, unsigned int n, unsigned int p)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
 
 	if (row < m && col < p)  
    { 
	    for(int i = 0; i < n; i++){
	    	//mC[row][col] += mA[row][i] * mB[i][col];
	    	mC[row][col] += 1;
	    }
    }
}

void cuda()
{
	    float **device_matrix_A = NULL;  
    float **host_matrix_A = NULL;  
    float *device_array_A = NULL;  
    float *host_array_A = NULL;  
    cudaError_t res;  
    int r, c;    
  
    res = cudaMalloc((void**)(&device_matrix_A), M_*sizeof(float*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_A), M_*N_*sizeof(float));CHECK(res)  
    host_matrix_A = (float**)malloc(M_*sizeof(float*));  
    host_array_A = (float*)malloc(M_*N_*sizeof(float));  
  
    for (r = 0; r < M_; r++)  
    {  
        host_matrix_A[r] = device_array_A + r*N_;  
    }  

    res = cudaMemcpy((void*)(device_matrix_A), (void*)(host_matrix_A), M_*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)  
    dim3 dimBlock(16,16);  
    dim3 dimGrid((N_+dimBlock.x-1)/(dimBlock.x), (M_+dimBlock.y-1)/(dimBlock.y));  
    InitMatrix<<<dimGrid, dimBlock>>>(device_matrix_A, M_, N_, 1);  
    res = cudaMemcpy((void*)(host_array_A), (void*)(device_array_A), M_*N_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < M_; r++)  
    {  
        for (c = 0; c < N_; c++)  
        {  
            printf("%.6f ", host_array_A[r*N_+c]);   
        }  
        printf("\n");  
    } 


    float **device_matrix_B = NULL;  
    float **host_matrix_B = NULL;  
    float *device_array_B = NULL;  
    float *host_array_B = NULL;  
  
    res = cudaMalloc((void**)(&device_matrix_B), N_*sizeof(float*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_B), N_*P_*sizeof(float));CHECK(res)  
    host_matrix_B = (float**)malloc(N_*sizeof(float*));  
    host_array_B = (float*)malloc(N_*P_*sizeof(float));  
  
    for (r = 0; r < N_; r++)  
    {  
        host_matrix_B[r] = device_array_B + r*P_;  
    }  

    res = cudaMemcpy((void*)(device_matrix_B), (void*)(host_matrix_B), N_*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)   
    InitMatrix<<<dimGrid, dimBlock>>>(device_matrix_B, N_, P_, 2);  
    res = cudaMemcpy((void*)(host_array_B), (void*)(device_array_B), N_*P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < N_; r++)  
    {  
        for (c = 0; c < P_; c++)  
        {  
            printf("%.6f ", host_array_B[r*P_+c]);   
        }  
        printf("\n");  
    }  


    float **device_matrix_C = NULL;  
    float **host_matrix_C = NULL;  
    float *device_array_C = NULL;  
    float *host_array_C = NULL;  

    res = cudaMalloc((void**)(&device_matrix_C), M_*sizeof(float*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_C), M_*P_*sizeof(float));CHECK(res)  
    host_matrix_C = (float**)malloc(M_*sizeof(float*));  
    host_array_C = (float*)malloc(M_*P_*sizeof(float));  

    for (r = 0; r < M_; r++)  
    {  
        host_matrix_C[r] = device_array_C + r*P_;  
    } 

    res = cudaMemcpy((void*)(device_matrix_C), (void*)(host_matrix_C), M_*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res) 
    Multiply<<<dimGrid, dimBlock>>>(device_matrix_A, device_matrix_B, device_matrix_C, M_, N_, P_);  
    res = cudaMemcpy((void*)(host_array_C), (void*)(device_array_C), M_*P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < M_; r++)  
    {  
        for (c = 0; c < P_; c++)  
        {  
            printf("%.6f ", host_array_C[r*P_+c]);   
        }  
        printf("\n");  
    }  

    cudaFree((void*)device_matrix_A);  
    cudaFree((void*)device_array_A);  
    cudaFree((void*)device_matrix_B);  
    cudaFree((void*)device_array_B);  
    cudaFree((void*)device_matrix_C);  
    cudaFree((void*)device_array_C); 
    free(host_matrix_A);  
    free(host_array_A);  
    free(host_matrix_B);  
    free(host_array_B); 
    free(host_matrix_C);  
    free(host_array_C); 
}  

void sequential()
{
	
}
int main(int argc, char **argv)  
{  
	cuda();
	sequential();
    return 0;  
}  