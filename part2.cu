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
    int **device_m = NULL;  
    int **host_m = NULL;  
    int *device_a = NULL;  
    int *host_a = NULL;  
    cudaError_t res;  
    int r, c;  
    bool is_right=true;  
  
    res = cudaMalloc((void**)(&device_m), M_*sizeof(int*));CHECK(res)  
    res = cudaMalloc((void**)(&device_a), M_*N_*sizeof(int));CHECK(res)  
    host_m = (int**)malloc(M_*sizeof(int*));  
    host_a = (int*)malloc(M_*N_*sizeof(int));  
  
    for (r = 0; r < M_; r++)  
    {  
        host_m[r] = device_a + r*N_;  
    }  
    res = cudaMemcpy((void*)(device_m), (void*)(host_m), M_*sizeof(int*), cudaMemcpyHostToDevice);CHECK(res)  
    dim3 dimBlock(16,16);  
    dim3 dimGrid((N_+dimBlock.x-1)/(dimBlock.x), (M_+dimBlock.y-1)/(dimBlock.y));  
    InitMatrix<<<dimGrid, dimBlock>>>(device_m, M_, N_, 1);  
    res = cudaMemcpy((void*)(host_a), (void*)(device_a), M_*N_*sizeof(int), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < M_; r++)  
    {  
        for (c = 0; c < N_; c++)  
        {  
            printf("%4d ", host_a[r*N_+c]);   
        }  
        printf("\n");  
    }  

    cudaFree((void*)device_m);  
    cudaFree((void*)device_a);  
    free(host_m);  
    free(host_a);  

    return 0;  
}  