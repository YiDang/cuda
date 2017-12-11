#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline static void __checkCudaErrors( cudaError err, const char *file, const int line )     {

    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

texture<int, cudaTextureType2D> tex_transition;

int main ( void ) {

    int m = 8, p_size = 100, alphabet = 20;

    size_t pitch;

    int *transition = ( int * ) malloc ( ( m * p_size + 1 ) * alphabet * sizeof ( int ) );
    memset ( transition, -1, ( m * p_size + 1 ) * alphabet * sizeof ( int ) );

    int *d_transition;

    checkCudaErrors ( cudaMallocPitch ( &d_transition, &pitch, alphabet * sizeof ( int ), ( m * p_size + 1 ) ) );

    checkCudaErrors ( cudaMemcpy2D ( d_transition, pitch, transition, alphabet * sizeof ( int ), alphabet * sizeof ( int ), ( m * p_size + 1 ), cudaMemcpyHostToDevice ) );

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    checkCudaErrors ( cudaBindTexture2D ( 0, tex_transition, d_transition, desc, alphabet * sizeof ( int ), ( m * p_size + 1 ), pitch ) );

    cudaFree ( d_transition );

    return 0;
}