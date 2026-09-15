// Name:Jacob Hininger
// Not simple Julia Set on the GPU
// nvcc jacobhw7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size. 
   I will pick these on the fly unsigned int WindowWidth, WindowHeight; 
   float XMin, XMax, YMin, YMax; and your code should work. You will be graded on this.
   
 - But you can set these values to whatever you want for the art compitition.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
 - Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.
 - This is a competition with a prize!!!
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

/*
 Explain what you did to fix the code:
 1. Lines 75-76: Fixed missing semicolons after "float cx = x" and "float cy = y"
 2. Line 83: Fixed missing semicolon after "tempY = y"
 3. Line 91: Fixed extra closing parenthesis in the return statement
 4. Line 108: Fixed "width"/"height" typo, changed to "WindowWidth"/"WindowHeight" in the bounds check
 5. Lines 53, 99, 168: Added WindowWidth and WindowHeight as parameters to colorPixels (prototype, definition, and the call that launches it), so it works for any window size, not just 1024x1024
 6. Lines 112-113: Fixed the pixel index calculation to use both row and col (added flippedRow, then used it in the id calculation) instead of just col, which was causing rows to overwrite each other
 7. Line 112: Flipped the image vertically by adding a flippedRow variable, since the picture was rendering upside down
 8. Lines 91, 95: Changed escapeOrNotColor to return a smooth/continuous value instead of a plain integer count, and return -1.0 for points that never escape, to reduce speckly/noisy colors at the edge
 9. Lines 128-134: Added a separate flat color for interior points (t < 0.0f) instead of running them through the same color formula as everything else
 10. Lines 139-140: Changed the green and blue color frequency numbers from 2.5 and 4.0 to 1.4 and 2.0 so the colors transition more smoothly
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
unsigned int WindowWidth = 1920;
unsigned int WindowHeight = 1080;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor (float, float);
__global__ void colorPixels(float, float, float, float, float, unsigned int, unsigned int);
void display(void);

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
	
	float cx = x;
	float cy = y;
	float tempY;
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		tempY = y;
		x = tempX*tempX -tempY*tempY +cx;
		y = (2.0 * fabsf(tempX) * fabsf(tempY)) + cy;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count == maxCount) 
	{
		return -1.0f;;
	}
	else
	{
		return count +1.0f - log2f(logf(mag) /logf(maxMag));
	}
}

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, unsigned int WindowWidth, unsigned int WindowHeight) 
{
	float x,y;
	int id;
	
	// 2-D pixel position
	int col = threadIdx.x +blockDim.x * blockIdx.x;
	int row = threadIdx.y +blockDim.y * blockIdx.y;

	if(col >= WindowWidth || row >= WindowHeight) return;

	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	int flippedRow = WindowHeight - 1 - row; // the image looked flipped so I added this
	id = 3*(flippedRow*WindowWidth + col);
	
	//Asigning each thread its x and y value of its pixel.
	x = xMin + dx*col;
	y = yMin + dy*row;

	/*float t = escapeOrNotColor(x,y) / (float)MAXITERATIONS;
	
	pixels[id]   = 0.5f + 0.5f*cosf(6.2832f*(1.0f*t + 0.00f)); // Red
    pixels[id+1] = 0.5f + 0.5f*cosf(6.2832f*(2.5f*t + 0.33f)); // Green
    pixels[id+2] = 0.5f + 0.5f*cosf(6.2832f*(4.0f*t + 0.67f)); // Blue
	*/

	float t = escapeOrNotColor(x,y); // this just makes it look better

	if (t < 0.0f)
	{
		// Interior of the set -- flat color instead of noisy cosine output
		pixels[id]   = 0.05f;
		pixels[id+1] = 0.0f;
		pixels[id+2] = 0.10f;
	}
	else
	{
		t = t / (float)MAXITERATIONS;
		pixels[id]   = 0.5f + 0.5f*cosf(6.2832f*(1.0f*t + 0.00f)); // Red
		pixels[id+1] = 0.5f + 0.5f*cosf(6.2832f*(1.4f*t + 0.33f)); // Green
		pixels[id+2] = 0.5f + 0.5f*cosf(6.2832f*(2.0f*t + 0.67f)); // Blue
	}
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	
	blockSize.x = 16; //WindowWidth;
	blockSize.y = 16;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x = (WindowWidth  + blockSize.x - 1) / blockSize.x;
    gridSize.y = (WindowHeight + blockSize.y - 1) / blockSize.y;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
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


