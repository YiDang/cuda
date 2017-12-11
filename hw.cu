#include <cstdio>
#include <iostream>


using namespace std;

texture<float,2,cudaReadModeElementType> tex_w;

__global__ void kernel(int imax, float (*f)[3])
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  // width = 3, height = imax                                                                         
  // but we have imax threads in x, 3 in y                                                            
  // therefore height corresponds to x threads (i)                                                    
  // and width corresponds to y threads (j)                                                           
  if(i<imax)
    {
      // linear filtering looks between indices                                                       
      f[i][j] = tex2D(tex_w, j+0.5f, i+0.5f);
    }
}

void print_to_stdio(int imax, float (*w)[3])
{
  for (int i=0; i<imax; i++)
    {
      printf("%2d  %3.3f  %3.3f  %3.3f\n",i, w[i][0], w[i][1], w[i][2]);
    }
  printf("\n");
}

int main(void)
{
  int imax = 8;
  float (*w)[3];
  float (*d_f)[3], *d_w;
  dim3 grid(imax,3);

  w = (float (*)[3])malloc(imax*3*sizeof(float));

  for(int i=0; i<imax; i++)
    {
      for(int j=0; j<3; j++)
        {
          w[i][j] = i + 0.01f*j;
        }
    }

  print_to_stdio(imax, w);

  size_t pitch;
	cudaMallocPitch((void**)&d_w, &pitch, 3*sizeof(float), imax);

   cudaMemcpy2D(d_w,             // device destination                                   
                             pitch,           // device pitch (calculated above)                      
                             w,               // src on host                                          
                             3*sizeof(float), // pitch on src (no padding so just width of row)       
                             3*sizeof(float), // width of data in bytes                               
                             imax,            // height of data                                       
                             cudaMemcpyHostToDevice) ;

  cudaBindTexture2D(NULL, tex_w, d_w, tex_w.channelDesc, 3, imax, pitch) ;

  tex_w.normalized = false;  // don't use normalized values                                           
  tex_w.filterMode = cudaFilterModeLinear;
  tex_w.addressMode[0] = cudaAddressModeClamp; // don't wrap around indices                           
  tex_w.addressMode[1] = cudaAddressModeClamp;

  // d_f will have result array                                                                       
  cudaMalloc( &d_f, 3*imax*sizeof(float) );

  // just use threads for simplicity                                                                  
  kernel<<<1,grid>>>(imax, d_f);

  cudaMemcpy(w, d_f, 3*imax*sizeof(float), cudaMemcpyDeviceToHost);

  cudaUnbindTexture(tex_w);
  cudaFree(d_w);
  cudaFree(d_f);

  print_to_stdio(imax, w);

  free(w);
  return 0;
}