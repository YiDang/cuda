#include <stdio.h>
 
const int N = 16; 
const int blocksize = 16; 

#define DIMX = 10000;
#define DIMY = 10000;

__global__ 
void hello(char *a, int *b) 
{
	
}
 
int main()
{
	float m[DIMX][DIMY];
	printf("%d\n", m[1][1]);
	return EXIT_SUCCESS;
}