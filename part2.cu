#include <stdio.h>  
#include <stdlib.h>  
#include <cuda_runtime.h>  
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <iostream>
#include <thrust/device_vector.h>
#define M_ 2  
#define N_ 6 
#define P_ 4 

#define BLOCK_SIZE 4
#define TILE_WIDTH 4;
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  

#define show(matrix, lenm, lenn) for(int r = 0; r < lenm; r++){for (int c = 0; c < lenn; c++){printf("%.6f ", matrix[r*lenn+c]);}printf("\n");}printf("\n");



__global__ void InitArray(float *a, unsigned int rows, unsigned int cols, int seed)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
    curandState_t state;
    
    if (row < rows && col < cols)  
    {  
        curand_init((row*cols + col) * seed, /* the seed controls the sequence of random values that are produced */
                  row, /* the sequence number is only important with multiple cores */
                  0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                  &state);

        //a[row][col] = curand_uniform(&state);
        a[row * cols + col] = row * cols + col;
    }  
}

__global__ void Multiply(float *arrayA, float *arrayB, float *arrayC, unsigned int m, unsigned int n, unsigned int p)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  
 	
    __shared__ float sA[M_*N_];
    __shared__ float sB[P_*N_];
    __shared__ float sC[M_*P_];
 	//copy A
    if (row < m && col < n) 
    {
        int idx = row * n +col;
        sA[idx] = arrayA[idx]; 
    }

    //copy B
    if (row < n && col < p) 
    {
        int idx = row * p +col;
        sB[idx] = arrayB[idx]; 
    }

    __syncthreads();
    //if(row == 0 && col ==0)
    //{
    //    for(int i = 0; i < m * n; i++)
    //    {
    //        printf("%2f,", sA[i]);
    //    }
    //    printf("\n");
//
    //    for(int i = 0; i < p * n; i++)
    //    {
    //        printf("%2f,", sB[i]);
    //    }
    //    printf("\n");
    //    
    //}

 	if (row < m && col < p)  
    { 
        int idx = row * p + col;

    	sC[idx] = 0;
	    for(int i = 0; i < n; i++)
        {
	    	sC[idx] += sA[row * n + i] * sB[i * p + col];
            //if(idx == 0 ) printf("sc[0] = %2f , sa[0] = %2f , sb[0] = %2f , idxa = %d, idxb = %d\n", sC[idx], sA[row * n + i] , sB[i * p + col], row * n + i, i * p + col);
	    }
        arrayC[idx] = sC[idx];
    }
}

__global__ void Multi(float *arrayA, float *arrayB, float *arrayC, unsigned int m, unsigned int n, unsigned int p)  
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.y * by + ty;  
    int col = blockDim.x * bx + tx;

    __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];

    float v = 0.0;

    for (int i = 0; i < (int)(ceil((float)n/TILE_WIDTH)); i++)
    {
        if (i*TILE_WIDTH + tx < n && row < m)
            sharedM[ty][tx] = arrayA[row*n + i*TILE_WIDTH + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i*TILE_WIDTH + ty < n && col < p)
            sharedN[ty][tx] = arrayB[(i*TILE_WIDTH + ty)*p + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; j++)
            v += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

    if (row < m && col < p)
        arrayC[row*p + col] = v;
}

    
void cudaInit(float *host_array_A, int rows, int cols)
{
    cudaError_t res;
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);  
    dim3 dimGrid((cols+dimBlock.x-1)/(dimBlock.x), (rows+dimBlock.y-1)/(dimBlock.y));

    float *device_array_A = NULL;
    res = cudaMalloc((void**)(&device_array_A), rows * cols * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_A), (void*)(host_array_A), rows * cols * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)
    InitArray<<<dimGrid, dimBlock>>>(device_array_A, rows, cols, 1);
    res = cudaMemcpy((void*)(host_array_A), (void*)(device_array_A), rows * cols * sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)  

    cudaFree((void*)device_array_A);
} 

void cudaMul(float *host_array_A, float *host_array_B, float *host_array_C)
{
    cudaError_t res;
     
    int maxd = std::max(P_ ,std::max(M_ , N_));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid((maxd+ dimBlock.x-1)/(dimBlock.x), (maxd + dimBlock.y-1)/(dimBlock.y));

    float *device_array_A = NULL;
    res = cudaMalloc((void**)(&device_array_A), M_ * N_ * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_A), (void*)(host_array_A), M_ * N_ * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    float *device_array_B = NULL;
    res = cudaMalloc((void**)(&device_array_B), N_ * P_ * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_B), (void*)(host_array_B), N_ * P_ * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    float *device_array_C = NULL;
    res = cudaMalloc((void**)(&device_array_C), M_ * P_ * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_C), (void*)(host_array_C), M_ * P_ * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    //Multiply<<<dimGrid, dimBlock>>>(device_array_A, device_array_B, device_array_C, M_, N_, P_);
    Multi<<<dimGrid, dimBlock>>>(device_array_A, device_array_B, device_array_C, M_, N_, P_);

    //res = cudaMemcpy((void*)(host_array_A), (void*)(device_array_A), M_ * N_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)
    //res = cudaMemcpy((void*)(host_array_B), (void*)(device_array_B), N_ * P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)
    res = cudaMemcpy((void*)(host_array_C), (void*)(device_array_C), M_ * P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)

}

void sequential(float *host_array_A, float *host_array_B, float *host_array_C)
{
	for(int i = 0; i < M_; i++)
	{
		for(int j = 0; j < P_; j++)
		{	
			host_array_C[i * P_ + j] = 0;
			for(int k = 0; k < N_; k++)
			{
				//printf("index%d\n", i * M_ + j);
				host_array_C[i * P_ + j] += host_array_A[i * N_ + k] * host_array_B[k * P_ + j];
				//printf("%2f,%2f,%2f,\n", host_array_A[i * N_ + k], host_array_B[k * P_ + j], host_array_C[i * M_ + j]);
			}
		}
	}
}

void cublas(float *host_array_A, float *host_array_B, float *host_array_C)
{
 	//show(host_array_A, M_, N_);
	//show(host_array_B, N_, P_);
	thrust::host_vector<float> hvA(M_ * N_);
	thrust::host_vector<float> hvB(N_ * P_);
	for(int i = 0; i < M_ * N_; i++) 
	{
		hvA[i] = host_array_A[i];
	}
	for(int i = 0; i < P_ * N_; i++) 
	{
		hvB[i] = host_array_B[i];
	}
	thrust::device_vector<float> dvA = hvA;
	thrust::device_vector<float> dvB = hvB;
	thrust::device_vector<float> dvC(M_ * P_);

	 // Do the actual multiplication

    int lda=N_ ,ldb=P_, ldc=P_;
    const float alpha = 1.0f;
    const float beta = 0.0f;
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "!!!! CUBLAS initialization error\n";
    }

    // Do the actual multiplication
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                            P_, M_, N_, 
                            &alpha, 
                            thrust::raw_pointer_cast(&dvB[0]), ldb, 
                            thrust::raw_pointer_cast(&dvA[0]), lda, 
                            &beta, 
                            thrust::raw_pointer_cast(&dvC[0]), ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "!!!! kernel execution error.\n";
    }

    for(int i = 0; i < M_; i++) 
	{
		for(int j = 0; j < P_; j++)
			std::cout<< dvC[i * P_ + j] <<" ";
		std::cout<< std::endl;
	}

    // Destroy the handle



}
int main(int argc, char **argv)  
{  
	float *host_array_A = (float*)malloc(M_*N_*sizeof(float)); 
	float *host_array_B = (float*)malloc(P_*N_*sizeof(float));
	float *host_array_C_para = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_seq = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_cublas = (float*)malloc(M_*P_*sizeof(float));

    cudaInit(host_array_A, M_, N_);
	show(host_array_A, M_, N_);
    cudaInit(host_array_B, N_, P_);
	show(host_array_B, N_, P_);

    printf("cuda start\n");
    cudaMul(host_array_A, host_array_B, host_array_C_para);
    //show(host_array_A, M_, N_);
    //show(host_array_B, N_, P_);
	show(host_array_C_para, M_, P_);

    printf("seq start\n");
	sequential(host_array_A, host_array_B, host_array_C_seq);
	show(host_array_C_seq, M_, P_);

    printf("cublas start\n");
    cublas(host_array_A, host_array_B, host_array_C_cublas);

    //show(host_array_C_cublas, M_, P_);
    
	free(host_array_A); 
	free(host_array_B); 
	free(host_array_C_para); 
	free(host_array_C_seq); 
	free(host_array_C_cublas); 
    return 0;  
}  