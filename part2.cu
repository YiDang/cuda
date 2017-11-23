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
__global__ void InitMatrix(int **da, unsigned int rows, unsigned int cols, int seed)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
    curandState_t state;
    curand_init((row*cols + col) * seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);
    if (row < rows && col < cols)  
    {  
        da[row][col] = curand(&state) % MAX;;  
    }  
}  
  
int main(int argc, char **argv)  
{  
    int **device_matrix_A = NULL;  
    int **host_matrix_A = NULL;  
    int *device_array_A = NULL;  
    int *host_array_A = NULL;  
    cudaError_t res;  
    int r, c;  
    bool is_right=true;  
  
    res = cudaMalloc((void**)(&device_matrix_A), M_*sizeof(int*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_A), M_*N_*sizeof(int));CHECK(res)  
    host_matrix_A = (int**)malloc(M_*sizeof(int*));  
    host_array_A = (int*)malloc(M_*N_*sizeof(int));  
  
    for (r = 0; r < M_; r++)  
    {  
        host_matrix_A[r] = device_array_A + r*N_;  
    }  
    res = cudaMemcpy((void*)(device_matrix_A), (void*)(host_matrix_A), M_*sizeof(int*), cudaMemcpyHostToDevice);CHECK(res)  
    dim3 dimBlock(16,16);  
    dim3 dimGrid((N_+dimBlock.x-1)/(dimBlock.x), (M_+dimBlock.y-1)/(dimBlock.y));  
    InitMatrix<<<dimGrid, dimBlock>>>(device_matrix_A, M_, N_, 1);  
    res = cudaMemcpy((void*)(host_array_A), (void*)(device_array_A), M_*N_*sizeof(int), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < M_; r++)  
    {  
        for (c = 0; c < N_; c++)  
        {  
            printf("%4d ", host_array_A[r*N_+c]);   
        }  
        printf("\n");  
    } 


    int **device_matrix_B = NULL;  
    int **host_matrix_B = NULL;  
    int *device_array_B = NULL;  
    int *host_array_B = NULL;  
    is_right=true;  
  
    res = cudaMalloc((void**)(&device_matrix_B), M_*sizeof(int*));CHECK(res)  
    res = cudaMalloc((void**)(&device_array_B), M_*N_*sizeof(int));CHECK(res)  
    host_matrix_B = (int**)malloc(M_*sizeof(int*));  
    host_array_B = (int*)malloc(M_*N_*sizeof(int));  
  
    for (r = 0; r < M_; r++)  
    {  
        host_matrix_B[r] = device_array_B + r*N_;  
    }  
    res = cudaMemcpy((void*)(device_matrix_B), (void*)(host_matrix_B), M_*sizeof(int*), cudaMemcpyHostToDevice);CHECK(res)   
    InitMatrix<<<dimGrid, dimBlock>>>(device_matrix_B, M_, N_, 2);  
    res = cudaMemcpy((void*)(host_array_B), (void*)(device_array_B), M_*N_*sizeof(int), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < M_; r++)  
    {  
        for (c = 0; c < N_; c++)  
        {  
            printf("%4d ", host_array_B[r*N_+c]);   
        }  
        printf("\n");  
    }  

    cudaFree((void*)device_matrix_A);  
    cudaFree((void*)device_array_A);  
    cudaFree((void*)device_matrix_B);  
    cudaFree((void*)device_array_B);  
    free(host_matrix_A);  
    free(host_array_A);  
    free(host_matrix_B);  
    free(host_array_B); 

    return 0;  
}  