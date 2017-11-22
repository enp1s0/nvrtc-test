/*
 * NVIDIA Runtime Compilation
 * 2017.11.22
 */

#include <iostream>
#include <nvrtc.h>
#include <cuda.h>
#include "cuda_common.h"

const int NUM_BLOCKS = 16;
const int NUM_THREADX = 16;

const char *kernel_code = "															\n\
						   extern \"C\"												\n\
						   __global__ void kernel(float *y,float *x,int max_tid){	\n\
							   int tid = blockIdx.x * blockDim.x + threadIdx.x;		\n\
							   if(tid >= max_tid)return;							\n\
							   y[tid] = __sinf(x[tid]);								\n\
						   }														\n";

int main(){
	nvrtcProgram program;
	nvrtcCreateProgram(&program,
			kernel_code,
			"kernel.cu",
			0,
			NULL,
			NULL);
	const char *options[] = {
		"--gpu-architecture=compute_60",
	};
	nvrtcResult result = nvrtcCompileProgram(program,
			1,
			options);
	size_t log_size;
	nvrtcGetProgramLogSize(program,&log_size);
	char *log = new char[log_size];
	nvrtcGetProgramLog(program,log);
	std::cout<<log<<std::endl;
	delete [] log;
	if(result != NVRTC_SUCCESS){
		std::cerr<<"Compilation failed"<<std::endl;
		return 1;
	}
	size_t ptx_size;
	nvrtcGetPTXSize(program,&ptx_size);
	char *ptx = new char [ptx_size];
	nvrtcGetPTX(program,ptx);
	nvrtcDestroyProgram(&program);

	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction cuFunction;
	cuInit(0);
	cuDeviceGet(&cuDevice,0);
	cuCtxCreate(&cuContext,0,cuDevice);
	cuModuleLoadDataEx(&cuModule,ptx,0,0,0);
	cuModuleGetFunction(&cuFunction,cuModule,"kernel");

	size_t n = NUM_BLOCKS * NUM_THREADX;
	size_t mem_size = sizeof(float) * n;
	float *dx,*dy;
	float *hx,*hy;

	CUDA_HANDLE_ERROR( cudaMalloc((void**)&dx,mem_size) );
	CUDA_HANDLE_ERROR( cudaMalloc((void**)&dy,mem_size) );
	CUDA_HANDLE_ERROR( cudaMallocHost((void**)&hx,mem_size) );
	CUDA_HANDLE_ERROR( cudaMallocHost((void**)&hy,mem_size) );
	for(int i = 0;i < n;i++)hx[i] = i* 3.141592f/n;
	CUDA_HANDLE_ERROR( cudaMemcpy(dx,hx,mem_size,cudaMemcpyHostToDevice) );

	void *args[] = {&dy,&dx,&n};
	cuLaunchKernel(cuFunction,
				NUM_BLOCKS,1,1,
				NUM_THREADX,1,1,
				0,NULL,
				args,0);
	cuCtxSynchronize();
	
	CUDA_HANDLE_ERROR( cudaMemcpy(hy,dy,mem_size,cudaMemcpyDeviceToHost) );
	for(int i = 0;i < n;i++){
		std::cout<<"sin( "<<hx[i]<<" ) = "<<hy[i]<<" ;";
		if((i+1)%6 == 0)std::cout<<std::endl;
	}

	CUDA_HANDLE_ERROR( cudaFree(dx) );
	CUDA_HANDLE_ERROR( cudaFree(dy) );
	CUDA_HANDLE_ERROR( cudaFreeHost(hx) );
	CUDA_HANDLE_ERROR( cudaFreeHost(hy) );

	delete [] ptx;
	cuModuleUnload(cuModule);
	cuCtxDestroy(cuContext);
}
