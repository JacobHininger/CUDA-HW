// Name:Jacob Hininger
// Device query
// nvcc Jacobhw5 -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
 added line 81
 added line 82
 added an explanation of what each piece of information means
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i); // tells which GPU it's talking about
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name); //Name of the GPU
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); //Compute capability of the GPU
		printf("Clock rate: %d\n", prop.clockRate); //Clock rate of the GPU
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n"); //Whether the device can concurrently copy memory and execute a kernel
		else printf("Disabled\n");
		printf("Kernel execution timeout : "); //Whether there is a run time limit on kernels
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i); //Memory information for the GPU
		printf("Total global mem: %ld\n", prop.totalGlobalMem); //Total amount of global memory on the GPU
		printf("Total constant Mem: %ld\n", prop.totalConstMem); //Total amount of constant memory on the GPU
		printf("Max mem pitch: %ld\n", prop.memPitch); // Maximum pitch allowed for memory copies, pitch is the number of bytes between rows in 2D memory copies
		printf("Texture Alignment: %ld\n", prop.textureAlignment); //Alignment requirement for textures
		printf(" ---MP Information for device %d ---\n", i); //Multiprocessor information for the GPU
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock); //Amount of shared memory available per block
		printf("Registers per mp: %d\n", prop.regsPerBlock); //Number of 32-bit registers available per block
		printf("Threads in warp: %d\n", prop.warpSize); //Number of threads in a warp, a warp is a group of threads that execute the same instruction at the same time
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); //Maximum number of threads that can be in a block
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //Maximum dimensions of a block
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); //Maximum dimensions of a grid
		printf("Memory bus width: %d bits\n", prop.memoryBusWidth); //Width of the memory bus in bits
		printf("Memory clock rate: %d kHz\n", prop.memoryClockRate); //Memory clock rate in kHz
		printf("\n"); 
		
	}	
	return(0);
}

