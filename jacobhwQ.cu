// Name:
// Vector addition on two GPUs.
// nvcc jacobhwQ.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 0:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 1:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

/*
 Purpose:
  To learn how to use more than one GPU.
*/

/*
 Explain what you did to fix the code:
 
 1. Added a GPU count check with cudaGetDeviceCount() in setUpDevices(). If fewer than two GPUs
    are found, the program prints a message and exits.
 2. Split the vector in two. The first GPU gets N/2 elements (integer division, rounds down) and the
    second GPU gets N - N/2 elements, so when N is odd the extra last element goes to GPU 1 and nothing
    is dropped. (N = 11503 -> 5751 and 5752.)
 3. Made two sets of GPU pointers (A_GPU0/B_GPU0/C_GPU0 and A_GPU1/B_GPU1/C_GPU1). Memory on a GPU
    belongs to whichever device was active when cudaMalloc was called, so I call cudaSetDevice(0) or
    cudaSetDevice(1) before every allocation, copy, kernel launch, and free for that GPU.
 4. GPU 0 works on A_CPU[0 .. half0-1] and GPU 1 works on A_CPU[half0 .. N-1]. I do this by passing
    the host pointers offset by half0 (A_CPU + half0) when copying to and from GPU 1.
 5. Each GPU gets its own grid size, computed from its own half length.
 6. Both kernels and copies are issued asynchronously, and then I call cudaSetDevice + cudaDeviceSynchronize
    on each GPU so the CPU waits for both to finish before checking the answer. This lets the two GPUs
    run at the same time.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU0, *B_GPU0, *C_GPU0; //GPU 0 pointers
float *A_GPU1, *B_GPU1, *C_GPU1; //GPU 1 pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize0; //Grid for GPU 0
dim3 GridSize1; //Grid for GPU 1
int HalfN0; // Number of elements handled by GPU 0
int HalfN1; // Number of elements handled by GPU 1
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float*, float*, float*, int);
bool  check(float*, int, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	// Making sure we have at least two GPUs.
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(deviceCount < 2)
	{
		printf("\n This program needs at least 2 GPUs but only %d was found. Exiting.\n\n", deviceCount);
		exit(0);
	}
	
	// Splitting the vector. If N is odd, GPU 1 gets the extra element so nothing is left out.
	HalfN0 = N/2;
	HalfN1 = N - HalfN0;
	
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize0.x = (HalfN0 - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize0.y = 1;
	GridSize0.z = 1;
	
	GridSize1.x = (HalfN1 - 1)/BlockSize.x + 1;
	GridSize1.y = 1;
	GridSize1.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU 0" Memory
	cudaSetDevice(0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A_GPU0, HalfN0*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU0, HalfN0*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU0, HalfN0*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Device "GPU 1" Memory
	cudaSetDevice(1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A_GPU1, HalfN1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU1, HalfN1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU1, HalfN1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = fabs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < tolerence) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	// Each GPU's memory must be freed while that GPU is the active device.
	cudaSetDevice(0);
	cudaFree(A_GPU0); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU0); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);
	cudaFree(A_GPU1); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU1); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU1);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPUs (also checks that we have two).
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPUs
	gettimeofday(&start, NULL);
	
	// ---- GPU 0: first half (elements 0 .. HalfN0-1) ----
	cudaSetDevice(0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(A_GPU0, A_CPU, HalfN0*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU0, B_CPU, HalfN0*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	addVectorsGPU<<<GridSize0,BlockSize>>>(A_GPU0, B_GPU0, C_GPU0, HalfN0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpyAsync(C_CPU, C_GPU0, HalfN0*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ---- GPU 1: second half (elements HalfN0 .. N-1) ----
	cudaSetDevice(1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(A_GPU1, A_CPU + HalfN0, HalfN1*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU1, B_CPU + HalfN0, HalfN1*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	addVectorsGPU<<<GridSize1,BlockSize>>>(A_GPU1, B_GPU1, C_GPU1, HalfN1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpyAsync(C_CPU + HalfN0, C_GPU1, HalfN1*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the CPU waits for BOTH GPUs to finish.
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPUs");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPUs was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
