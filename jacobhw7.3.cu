// Name:Jacob Hininger
// Not simple Julia Set on the GPU
// nvcc jacobhw7 (1).cu -o temp -lglut -lGL

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
 1. Lines 84-85: Fixed missing semicolons after "float cx = x" and "float cy = y"
 2. Line 92: Fixed missing semicolon after "tempY = y"
 3. Line 100: Fixed extra closing parenthesis in the return statement
 4. Line 117: Fixed "width"/"height" typo, changed to "WindowWidth"/"WindowHeight" in the bounds check
 5. Lines 62, 108, 177: Added WindowWidth and WindowHeight as parameters to colorPixels (prototype, definition, and the call that launches it), so it works for any window size, not just 1024x1024
 6. Lines 121-122: Fixed the pixel index calculation to use both row and col (added flippedRow, then used it in the id calculation) instead of just col, which was causing rows to overwrite each other
 7. Line 121: Flipped the image vertically by adding a flippedRow variable, since the picture was rendering upside down
 8. Lines 100, 104: Changed escapeOrNotColor to return a smooth/continuous value instead of a plain integer count, and return -1.0 for points that never escape, to reduce speckly/noisy colors at the edge
 9. Lines 137-143: Added a separate flat color for interior points (t < 0.0f) instead of running them through the same color formula as everything else
 10. Lines 148-149: Changed the green and blue color frequency numbers from 2.5 and 4.0 to 1.4 and 2.0 so the colors transition more smoothly
 11. Added AA define and supersampling in colorPixels: the "flame" filaments near the top of the burning ship are thinner than a single pixel, so sampling once per pixel there causes aliasing (pixels randomly flipping between escaped/not-escaped colors, which looks like noise). Now each pixel is sampled on an AA x AA grid (9 samples with AA=3) and the resulting colors are averaged, which smooths that region out.
 12. Added mouseClick() + glutMouseFunc registration: left-click zooms in (centered on the clicked point), right-click zooms out, by shrinking/growing XMin/XMax/YMin/YMax and calling glutPostRedisplay(). Also added free(pixelsCPU)/cudaFree(pixelsGPU) at the end of display(), since display() now runs many times instead of once and the buffers were never being released.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  10.824	//Real part of C
#define B  10.1711	//Imaginary part of C

// The "flame" filaments at the top of the burning ship are thinner than a
// single pixel at this resolution, so sampling once per pixel makes that
// area look noisy/speckled (aliasing). AA is how many samples we take per
// pixel along each axis (AA x AA samples total, averaged together) to
// smooth that out. 3 is a good quality/speed tradeoff; try 4-5 if you want
// it even smoother and don't mind the extra render time.
#define AA 3

// Global variables
unsigned int WindowWidth = 1920;
unsigned int WindowHeight = 1080;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  1.0;

// Zoom factors used by mouseClick() below. < 1.0 shrinks the visible
// range (zoom in), > 1.0 grows it (zoom out).
#define ZOOM_IN_FACTOR  0.5f
#define ZOOM_OUT_FACTOR 2.0f

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor (float, float);
__global__ void colorPixels(float, float, float, float, float, unsigned int, unsigned int);
void display(void);
void mouseClick(int, int, int, int);

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
		return count +1.0f - log2f(logf(mag) /logf(maxMag)); //this basically smooths out the color. 
	}
}

// Turns an escapeOrNotColor() result into an RGB triple. Pulled out into
// its own function so it can be called once per sub-pixel sample below.
__device__ void colorAt(float t, float *r, float *g, float *b)
{
	if (t < 0.0f)
	{
		// Interior of the set -- flat color instead of noisy cosine output
		*r = 0.05f;
		*g = 0.10f;
		*b = 0.10f;
	}
	else
	{
		t = t / (float)MAXITERATIONS;
		*r = 0.8f + 0.5f*cosf(6.2832f*(1.0f*t + 0.33f)); // Red
		*g = 0.5f + 0.5f*cosf(6.2832f*(1.4f*t + .75f)); // Green
		*b = 0.3f + 0.5f*cosf(6.2832f*(2.0f*t + 1.0f)); // Blue
	}
}

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, unsigned int WindowWidth, unsigned int WindowHeight) 
{
	int id;
	
	// 2-D pixel position
	int col = threadIdx.x +blockDim.x * blockIdx.x;
	int row = threadIdx.y +blockDim.y * blockIdx.y;

	if(col >= WindowWidth || row >= WindowHeight) return;

	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	int flippedRow = WindowHeight - 1 - row; // the image looked flipped so I added this
	id = 3*(flippedRow*WindowWidth + col);

	/*float t = escapeOrNotColor(x,y) / (float)MAXITERATIONS;
	
	pixels[id]   = 0.5f + 0.5f*cosf(6.2832f*(1.0f*t + 0.00f)); // Red
    pixels[id+1] = 0.5f + 0.5f*cosf(6.2832f*(2.5f*t + 0.33f)); // Green
    pixels[id+2] = 0.5f + 0.5f*cosf(6.2832f*(4.0f*t + 0.67f)); // Blue
	*/

	// Supersampling: the burning ship's "flame" filaments near the top of
	// the image are thinner than a pixel, so one sample per pixel makes
	// that area look noisy. We take an AA x AA grid of samples spread
	// across the pixel's footprint and average their colors instead.
	float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
	for (int sy = 0; sy < AA; sy++)
	{
		for (int sx = 0; sx < AA; sx++)
		{
			float x = xMin + dx*(col + (sx + 0.5f)/AA);
			float y = yMin + dy*(row + (sy + 0.5f)/AA);

			float t = escapeOrNotColor(x,y); // this just makes it look better
			float r,g,b;
			colorAt(t, &r, &g, &b);
			rSum += r;
			gSum += g;
			bSum += b;
		}
	}

	float numSamples = (float)(AA*AA);
	pixels[id]   = rSum/numSamples;
	pixels[id+1] = gSum/numSamples;
	pixels[id+2] = bSum/numSamples;
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

	// display() now gets called again every time we zoom, so we need to
	// free these or we leak a full frame's worth of memory on every click.
	free(pixelsCPU);
	cudaFree(pixelsGPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

void mouseClick(int button, int state, int x, int y)
{
	// Only act when a button is pressed, not released.
	if(state != GLUT_DOWN) return;

	float zoom;
	if(button == GLUT_LEFT_BUTTON)       zoom = ZOOM_IN_FACTOR;  // zoom in
	else if(button == GLUT_RIGHT_BUTTON) zoom = ZOOM_OUT_FACTOR; // zoom out
	else                                 return;                // ignore other buttons

	// Convert the clicked pixel into fractal-space coordinates. GLUT gives
	// (x, y) with (0,0) at the top-left of the window, which lines up with
	// how row/col map to x/y inside the kernel (row 0 = top of screen).
	float fx = XMin + (XMax - XMin) * ((float)x / (float)WindowWidth);
	float fy = YMin + (YMax - YMin) * ((float)y / (float)WindowHeight);

	float newWidth  = (XMax - XMin) * zoom;
	float newHeight = (YMax - YMin) * zoom;

	// Re-center the view on the clicked point at the new zoom level.
	XMin = fx - newWidth  / 2.0f;
	XMax = fx + newWidth  / 2.0f;
	YMin = fy - newHeight / 2.0f;
	YMax = fy + newHeight / 2.0f;

	glutPostRedisplay(); // ask GLUT to call display() again with the new bounds
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMouseFunc(mouseClick); // left click = zoom in, right click = zoom out
   	glutMainLoop();
}


