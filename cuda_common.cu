#include <stdio.h>
#include <stdlib.h>

void cudaHandleError( cudaError_t err,const char *file,int line ) {
	if (err != cudaSuccess) {
		printf( "CUDA Error\n%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
		exit( EXIT_FAILURE );
	}
}
