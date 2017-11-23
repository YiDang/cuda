#include <stdio.h>  
#include <stdlib.h>  
#include <cuda_runtime.h>  
  
#define ROWS 32  
#define COLS 16  
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  
__global__ void Kerneltest(int **da, unsigned int rows, unsigned int cols)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
    if (row < rows && col < cols)  
    {  
        da[row][col] = row*cols + col;  
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
  
    res = cudaMalloc((void**)(&da), ROWS*sizeof(int*));CHECK(res)  
    res = cudaMalloc((void**)(&dc), ROWS*COLS*sizeof(int));CHECK(res)  
    ha = (int**)malloc(ROWS*sizeof(int*));  
    hc = (int*)malloc(ROWS*COLS*sizeof(int));  
  
    for (r = 0; r < ROWS; r++)  
    {  
        ha[r] = dc + r*COLS;  
    }  
    res = cudaMemcpy((void*)(da), (void*)(ha), ROWS*sizeof(int*), cudaMemcpyHostToDevice);CHECK(res)  
    dim3 dimBlock(16,16);  
    dim3 dimGrid((COLS+dimBlock.x-1)/(dimBlock.x), (ROWS+dimBlock.y-1)/(dimBlock.y));  
    Kerneltest<<<dimGrid, dimBlock>>>(da, ROWS, COLS);  
    res = cudaMemcpy((void*)(hc), (void*)(dc), ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);CHECK(res)  
  
    for (r = 0; r < ROWS; r++)  
    {  
        for (c = 0; c < COLS; c++)  
        {  
            printf("%4d ", hc[r*COLS+c]);  
            if (hc[r*COLS+c] != (r*COLS+c))  
            {  
                is_right = false;  
            }  
        }  
        printf("\n");  
    }  
    printf("the result is %s!\n", is_right? "right":"false");  
    cudaFree((void*)da);  
    cudaFree((void*)dc);  
    free(ha);  
    free(hc);  
    getchar();  
    return 0;  
}  