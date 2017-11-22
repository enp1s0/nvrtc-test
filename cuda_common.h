#pragma once

#define CUDA_HANDLE_ERROR( err ) (cudaHandleError( err, __FILE__, __LINE__ ))

void cudaHandleError( cudaError_t err,const char *file,int line );

