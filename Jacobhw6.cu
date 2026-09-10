// Name: Jacob Hininger
// Simple Julia CPU.
// nvcc Jacobhw6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:
 1. Changed escapeOrNotColor() to a device function so it can run on the GPU.
 2. Added a global juliaKernel() function to calculate the pixels of the Julia fractal on the GPU.
 3. Changed the pixel calculation from the original CPU while loops to GPU threads, with each thread calculating one pixel.
 4. Added cudaMalloc() to allocate memory on the GPU for the pixel data.
 5. Added a CUDA kernel launch using blocks and threads to run juliaKernel() on the GPU.
 6. Added cudaMemcpy() to copy the calculated pixel data from the GPU back to the CPU.
 7. Added cudaFree() to free the GPU memory after the calculation is complete.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float, float);

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

__device__ float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

__global__ void juliaKernel(float *pixels, int width, int height, float xmin, float xmax, float ymin, float ymax)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
	int totalPixels = width * height;
    if (k >= totalPixels)
        return;
    int pixelX = k % width;
	int pixelY = k / width;
    float stepSizeX = (xmax - xmin) / ((float)width);
	float stepSizeY = (ymax - ymin) / ((float)height);
    float x = xmin + pixelX * stepSizeX;
	float y = ymin + pixelY * stepSizeY;
    pixels[k * 3] = escapeOrNotColor(x, y);
    pixels[k * 3 + 1] = 0.0;
    pixels[k * 3 + 2] = 0.0;
}

void display(void) 
{ 
	float *pixels; 
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);	
	float *dev_pixels;
    cudaMalloc((void**)&dev_pixels, WindowWidth * WindowHeight * 3 * sizeof(float));
    int totalPixels = WindowWidth * WindowHeight;
    int threadsPerBlock = 256;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    juliaKernel<<<blocks, threadsPerBlock>>>(dev_pixels, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(pixels, dev_pixels, WindowWidth * WindowHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(dev_pixels);    
    //Putting pixels on the screen.
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
    glFlush();  
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

