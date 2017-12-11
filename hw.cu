// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
 
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
int main()
{
	int  IE2D[4][5];
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<5;j++)
		IE2D[i][j] = i*5+j;
	}
	texture<int, 2>  texIE2D;

	int *dev_IE2D;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();

	cudaMallocPitch(  (void**) &dev_IE2D,   &pitch,   sizeof(int) * 5,   4  );

	cudaBindTexture2D( NULL,   texIE2D,   dev_IE2D,   desc,   5,   4,   pitch );

	for(int row = 0; row < 4; ++row) 
        cudaMemcpy(dev_IE2D[row*(pitch/sizeof(int))],   &IE2D[row][0],   sizeof(int)*5,   cudaMemcpyHostToDevice);
	
	dim3 dimBlock( blocksize, 16 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( bd );
	
	printf("%s\n", a);
	return EXIT_SUCCESS;
}