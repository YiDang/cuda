#include <stdio.h>  
#include <stdlib.h>  
#include <cuda_runtime.h>  
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <iostream>
#include <thrust/device_vector.h>

#define M_ 5
#define N_ 5
#define P_ 5

#define BLOCK_SIZE 32
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  

#define show(matrix, lenm, lenn) for(int r = 0; r < lenm; r++){for (int c = 0; c < lenn; c++){printf("%.6f ", matrix[r*lenn+c]);}printf("\n");}printf("\n");

uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

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

        //a[row * cols + col] = curand_uniform(&state);
        a[row * cols + col] = row * cols + col;
    }  
}

__global__ void Multiply(float *arrayA, float *arrayB, float *arrayC)  
{  
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;  
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;  

 	if (row < M_ && col < P_)  
    { 	
    	#pragma unroll
	    for(int i = 0; i < N_; i++)
        {
	    	arrayC[row * P_ + col] += arrayA[row * N_ + i] * arrayB[i * P_ + col];
	    }
    }
}


texture<float, 1, cudaReadModeElementType> texA;
texture<float, 1, cudaReadModeElementType> texB;
__global__ void MultiplyTexture(float *arrayC)  
{  

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (x < M_ && y < P_)
    {
        float a = 0, b = 0;
        float temp_result = 0;
        for (int i = 0; i < N_; i++)
        {
            a = tex1Dfetch(tex_A, y, x);
            //a = tex2D<float>(texA, y, x);
            
            printf("idx:%d,%d,v:%f\n",y,x,a);
            //b = tex1Dfetch(texB, i * P_ + x);
            temp_result += a * b;
        }
        arrayC[offset] = temp_result;
    }
}

__global__ void Multi_SM(float *arrayA, float *arrayB, float *arrayC)  
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.y * by + ty;  
    int col = blockDim.x * bx + tx;

    __shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

    float v = 0.0;
    #pragma unroll
    for (int i = 0; i < (int)(ceil((float)N_/BLOCK_SIZE)); i++)
    {
        if (i*BLOCK_SIZE + tx < N_ && row < M_)
            sharedM[ty][tx] = arrayA[row*N_ + i*BLOCK_SIZE + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i*BLOCK_SIZE + ty < N_ && col < P_)
            sharedN[ty][tx] = arrayB[(i*BLOCK_SIZE + ty)*P_ + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();
        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; j++)
            v += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

    if (row < M_ && col < P_)
        arrayC[row*P_ + col] = v;
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

double cudaMul(float *host_array_A, float *host_array_B, float *host_array_C, int method)
{	
    cudaError_t res;
     
    int maxd = std::max(P_ ,std::max(M_ , N_));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid((M_ + dimBlock.x-1)/(dimBlock.x), (P_ + dimBlock.y-1)/(dimBlock.y));

    float *device_array_A = NULL;
    res = cudaMalloc((void**)(&device_array_A), M_ * N_ * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_A), (void*)(host_array_A), M_ * N_ * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    float *device_array_B = NULL;
    res = cudaMalloc((void**)(&device_array_B), N_ * P_ * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_B), (void*)(host_array_B), N_ * P_ * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    float *device_array_C = NULL;
    res = cudaMalloc((void**)(&device_array_C), M_ * P_ * sizeof(float));CHECK(res)
    res = cudaMemcpy((void*)(device_array_C), (void*)(host_array_C), M_ * P_ * sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    double start = rdtsc();
    if(method == 0)
    {
    	Multiply<<<dimGrid, dimBlock>>>(device_array_A, device_array_B, device_array_C);
    }  
    else if(method == 1)
    {
    	Multi_SM<<<dimGrid, dimBlock>>>(device_array_A, device_array_B, device_array_C);
    }
    //else if(method == 2)
    //{
    //	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); 
    //	cudaBindTexture(NULL, texA, device_array_A, desc, M_ * N_ * sizeof(float));
	//	cudaBindTexture(NULL, texB, device_array_B, desc, N_ * P_ * sizeof(float));
	//	MultiplyTexture<<<dimGrid, dimBlock>>>(device_array_C);
    //}
    cudaThreadSynchronize();
    double end = rdtsc();

    res = cudaMemcpy((void*)(host_array_C), (void*)(device_array_C), M_ * P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)

    cudaFree((void*)device_array_A);
    cudaFree((void*)device_array_B);
    cudaFree((void*)device_array_C);
    
    return end - start;
}

texture<float,2,cudaReadModeElementType> tex_A;
texture<float,2,cudaReadModeElementType> tex_B;
double cudaMulTex(float *host_array_A, float *host_array_B, float *host_array_C)
{   
    cudaError_t res;
     
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid((M_ + dimBlock.x-1)/(dimBlock.x), (P_ + dimBlock.y-1)/(dimBlock.y));

    float (*d_a)[N_];
    float (*tmp)[N_];

    for(int i = 0; i < M_ ; i++)
    {
        for(int j = 0; j < N_; j++)
        {
            tmp[i][j] = host_array_A[i * N_ + j];
            print("%d ",tmp[i][j]);
        }
        print("\n");
    }
    size_t pitch;
    cudaMallocPitch((void**)&d_a, &pitch, N_*sizeof(float), M_);

    cudaMemcpy2D(d_a,             // device destination                                   
                             pitch,           // device pitch (calculated above)                      
                             tmp,               // src on host                                          
                             N_*sizeof(float), // pitch on src (no padding so just width of row)       
                             N_*sizeof(float), // width of data in bytes                               
                             M_,            // height of data                                       
                             cudaMemcpyHostToDevice) ;
    cudaBindTexture2D(NULL, tex_A, d_w, tex_A.channelDesc, N_, M_, pitch) ;

    tex_A.normalized = false;  // don't use normalized values                                           
    tex_A.filterMode = cudaFilterModeLinear;
    tex_A.addressMode[0] = cudaAddressModeClamp; // don't wrap around indices                           
    tex_A.addressMode[1] = cudaAddressModeClamp;

    //..........................

    double start = rdtsc();

    MultiplyTexture<<<dimGrid, dimBlock>>>(device_array_C);

    //cudaThreadSynchronize();
    double end = rdtsc();

    //res = cudaMemcpy((void*)(host_array_C), (void*)(device_array_C), M_ * P_*sizeof(float), cudaMemcpyDeviceToHost);CHECK(res)

    //cudaFree((void*)device_array_A);
    
    return end - start;
}

double sequential(float *host_array_A, float *host_array_B, float *host_array_C)
{	
	double start = rdtsc();
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
	double end = rdtsc();
	return end - start;
}

double cublas(float *host_array_A, float *host_array_B, float *host_array_C)
{

	thrust::host_vector<float> hvA(M_ * N_);
	thrust::host_vector<float> hvB(N_ * P_);
	thrust::host_vector<float> hvC(M_ * P_);
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
    double start = rdtsc();
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                            P_, M_, N_, 
                            &alpha, 
                            thrust::raw_pointer_cast(&dvB[0]), ldb, 
                            thrust::raw_pointer_cast(&dvA[0]), lda, 
                            &beta, 
                            thrust::raw_pointer_cast(&dvC[0]), ldc);

    double end = rdtsc();
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "!!!! kernel execution error.\n";
    }
    hvC = dvC;
    for(int i = 0; i < M_ * P_; i++) 
    {
    	host_array_C[i] = hvC[i];
    }

    // Destroy the handle
    cublasDestroy(handle);

    return end - start;
}

int main(int argc, char **argv)  
{  
	float *host_array_A = (float*)malloc(M_*N_*sizeof(float)); 
	float *host_array_B = (float*)malloc(P_*N_*sizeof(float));
	float *host_array_C_seq = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_cuda = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_tile = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_texture = (float*)malloc(M_*P_*sizeof(float));
	float *host_array_C_cublas = (float*)malloc(M_*P_*sizeof(float));

    int showma = 0, showdif = 0;
	double diff = 0;
    cudaInit(host_array_A, M_, N_);
	//show(host_array_A, M_, N_);
    cudaInit(host_array_B, N_, P_);
	//show(host_array_B, N_, P_);

	printf("cublas start\n");
    diff = 0;diff = cublas(host_array_A, host_array_B, host_array_C_cublas);
    if(showma) show(host_array_C_cublas, M_, P_);
    std::cout << "Time million cycles:\t\t"
            << static_cast<double>(diff) / (1024 * 1024)
            << std::endl<< std::endl;

    printf("cuda start\n");
    diff = 0;diff = cudaMul(host_array_A, host_array_B, host_array_C_cuda, 0);
	if(showma) show(host_array_C_cuda, M_, P_);
	std::cout << "Time million cycles:\t\t"
            << static_cast<double>(diff) / (1024 * 1024)
            << std::endl<< std::endl;
    double error = 0;
    for(int i = 0; i < M_ * N_; i++)
    {
    	double tmp = host_array_C_cublas[i] - host_array_C_cuda[i];
    	error += tmp * tmp;
        if(tmp != 0 && showdif)
        {
            printf("cuda:%f",tmp);
        }
    }
    if(showdif)std::cout << "error:\t\t"<< error << std::endl << std::endl;

	printf("cuda tiled start\n");
    diff = 0;diff = cudaMul(host_array_A, host_array_B, host_array_C_tile, 1);
	if(showma) show(host_array_C_tile, M_, P_);
	std::cout << "Time million cycles:\t\t"
            << static_cast<double>(diff) / (1024 * 1024)
            << std::endl<< std::endl;

    error = 0;
    for(int i = 0; i < M_ * N_; i++)
    {
    	double tmp = host_array_C_cublas[i] - host_array_C_tile[i];
    	error += tmp * tmp;
        if(tmp != 0 && showdif)
        {
            printf("tile:%f",tmp);
        }
    }
    if(showdif)std::cout << "error:\t\t"<< error << std::endl << std::endl;

    printf("cuda textured start\n");
    diff = 0;diff = cudaMulTex(host_array_A, host_array_B, host_array_C_texture, 2);
	 show(host_array_C_texture, M_, P_);
	std::cout << "Time million cycles:\t\t"
            << static_cast<double>(diff) / (1024 * 1024)
            << std::endl<< std::endl;

    error = 0;
    for(int i = 0; i < M_ * N_; i++)
    {
    	double tmp = host_array_C_cublas[i] - host_array_C_texture[i];
    	error += tmp * tmp;
        if(tmp != 0 && showdif)
        {
            printf("texture:%f",tmp);
        }
    }
    if(showdif)std::cout << "error:\t\t"<< error << std::endl << std::endl;

    printf("seq start\n");
	diff = 0;diff = sequential(host_array_A, host_array_B, host_array_C_seq);
	if(showma) show(host_array_C_seq, M_, P_);
	std::cout << "Time million cycles:\t\t"
            << static_cast<double>(diff) / (1024 * 1024)
            << std::endl<< std::endl;

   	error = 0;
    for(int i = 0; i < M_ * N_; i++)
    {
    	double tmp = host_array_C_cublas[i] - host_array_C_seq[i];
    	error += tmp * tmp;
        if(tmp != 0 && showdif)
        {
            printf("seq:%f,",tmp);
        }
    }
    if(showdif)std::cout << "error:\t\t"<< error << std::endl << std::endl;

	free(host_array_A); 
	free(host_array_B);  
	free(host_array_C_seq); 
	free(host_array_C_cuda); 
	free(host_array_C_tile); 
	free(host_array_C_cublas); 
	free(host_array_C_texture); 
	
    return 0;  
}  