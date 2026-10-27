// Name:
// GPU random walk. 
// nvcc jacobhwP.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Explain what you did to fix the code:
 I replaced the CPU rand() walk with a CUDA kernel where each of the 20 threads runs its own
 10,000-step walk. Each thread sets up its own cuRAND state with curand_init(), using a different
 seed per thread (base seed from time(NULL) plus the thread id). Inside the loop, curand_uniform()
 returns a float in (0, 1], and I step -1 if it is below 0.5 and +1 otherwise, for both X and Y.
 The final positions are stored in arrays allocated with cudaMallocManaged, so the CPU can read them
 directly after cudaDeviceSynchronize() with no cudaMemcpy. I also added error checking and freed
 the memory with cudaFree().
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>

// Defines
#define NUMBER_OF_WALKS 20

// Globals
int NumberOfRandomSteps = 10000;

// Function prototypes
__device__ int getRandomDirection(curandState*);
__global__ void randomWalk(int*, int*, int, unsigned long long);
void cudaErrorCheck(const char*, int);
int main(int, char**);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: %s = %s at line %d in file %s\n", cudaGetErrorName(error), cudaGetErrorString(error), line, file);
		exit(0);
	}
}

__device__ int getRandomDirection(curandState *state)
{	
	// curand_uniform returns a float in (0, 1].
	float randomNumber = curand_uniform(state);
	
	if(randomNumber < 0.5f) return(-1);
	else return(1);
}

__global__ void randomWalk(int *positionX, int *positionY, int numberOfSteps, unsigned long long seed)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(id < NUMBER_OF_WALKS)
	{
		// Each thread gets its own seed (seed + id) and its own generator state.
		curandState state;
		curand_init(seed + id, 0, 0, &state);
		
		int x = 0;
		int y = 0;
		for(int i = 0; i < numberOfSteps; i++)
		{
			x += getRandomDirection(&state);
			y += getRandomDirection(&state);
		}
		
		positionX[id] = x;
		positionY[id] = y;
	}
}

int main(int argc, char** argv)
{
	int *positionX, *positionY;
	
	// Unified memory: accessible from both the CPU and the GPU.
	cudaMallocManaged(&positionX, NUMBER_OF_WALKS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMallocManaged(&positionY, NUMBER_OF_WALKS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
	
	unsigned long long seed = (unsigned long long)time(NULL);
	
	// One block of 20 threads, one thread per walk.
	randomWalk<<<1, NUMBER_OF_WALKS>>>(positionX, positionY, NumberOfRandomSteps, seed);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Wait for the GPU to finish before the CPU touches the managed memory.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < NUMBER_OF_WALKS; i++)
	{
		printf("\n Walk %2d final position = (%d,%d) ", i, positionX[i], positionY[i]);
	}
	printf("\n");
	
	cudaFree(positionX);
	cudaFree(positionY);
	return 0;
}
