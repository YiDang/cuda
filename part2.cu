#include <stdio.h>
 
const int N = 16; 
const int blocksize = 16; 

#define ROWS = 10000;
#define COLS = 10000;

__global__ 
void hello(char *a, int *b) 
{
	
}
 
int main()
{
	int r, c;
	int **arr = (int**)malloc(ROWS*sizeof(int*));  
	int *data = (int*)malloc(COLS*ROWS*sizeof(int));  
	for (r = 0; r < ROWS; r++)  
	{  
	    arr[r] = data + r*COLS;  
	}  
  
	free(arr);  
	free(data)；
	return EXIT_SUCCESS;
}