#include <stdio.h>  
#include <stdlib.h>  
#include <cuda_runtime.h>  
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

#define M_ 32  
#define N_ 16 
#define P_ 16 

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
    int **da = NULL;  
    int **ha = NULL;  
    int *dc = NULL;  
    int *hc = NULL;  
    cudaError_t res;  
    int r, c;  
    bool is_right=true;  
  
    res = cudaMalloc((void**)(&da), M_*sizeof(int*));CHECK(res)  
    res = cudaMalloc((void**)(&dc), M_*N_*sizeof(int));CHECK(res)  
    ha = (int**)malloc(M_*sizeof(int*));  
    hc = (int*)malloc(M_*N_*sizeof(int));  
  
    for (r = 0; r < M_; r++)  
    {  
        ha[r] = dc + r*N_;  
    }  
    res = cudaMemcpy((void*)(da), (void*)(ha), M_*sizeof(int*), cudaMemcpyHostToDevice);CHECK(res)  
    dim3 dimBlock(16,16);  
    dim3 dimGrid((N_+dimBlock.x-1)/(dimBlock.x), (M_+dimBlock.y-1)/(dimBlock.y));  
    InitMatrix<<<dimGrid, dimBlock>>>(da, M_, N_, 1);  
    res = cudaMemcpy((void*)(hc), (void*)(dc), M_*N_*sizeof(int), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < M_; r++)  
    {  
        for (c = 0; c < N_; c++)  
        {  
            printf("%4d ", hc[r*N_+c]);   
        }  
        printf("\n");  
    }  

    cudaFree((void*)da);  
    cudaFree((void*)dc);  
    free(ha);  
    free(hc);  

    return 0;  
}  