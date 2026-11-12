// Name:
// nBody code on multiple GPUs. 
// nvcc T1_1GPU_To_2GPUs_Nbody.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code that runs on the GPU with all the bells and whistles removed. 
 Modify it so it runs on two GPUs. Don't use unified memory, physically copy the memory around
 to get an appreciation for unified memory and to see how you might want to do it this way for
 speed.
*/

/*
 Purpose:
 We know how to run on two GPUs to add vectors. Wow big deal! See it you can do it with something for complicated.
*/

/*
 Explain what you did to fix the code:
 
 I split the bodies in half. GPU 0 owns bodies [0, N/2) and GPU 1 owns bodies [N/2, N). Each GPU:
   - holds a FULL copy of the positions (PGPU) and masses (MGPU), because every body needs to "see"
     every other body when it computes forces.
   - holds velocity and force arrays (VGPU, FGPU) for ONLY its own half, since it only updates its own bodies.
 The kernels got an extra "offset" and "count" argument so each GPU only works on its own slice of the
 bodies (the inner j-loop still runs over all N bodies).
 After every time step each GPU only has fresh positions for its own half, so I synchronize both GPUs and
 swap the halves with cudaMemcpyPeer. After the swap both GPUs have all N updated positions again. 
 No unified memory is used anywhere; every cudaMalloc/cudaMemcpy is explicit. For drawing I copy the
 full position array back from GPU 0.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 10.0

// Globals
int N;
float3 *P, *V, *F;
float *M; 
float3 *PGPU[2], *VGPU[2], *FGPU[2]; // One set of pointers per GPU.
float *MGPU[2];
int Offset[2], Count[2];              // Which slice of the bodies each GPU owns.
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize[2];

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int);
void nBody();
int main(int, char**);

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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// Both GPUs hold all the positions after the exchange, so just grab them from GPU 0.
	cudaSetDevice(0);
	cudaMemcpy(P, PGPU[0], N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	N = 1000;
    	
    	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// Split the bodies between the two GPUs (works for odd N too).
	Count[0]  = N/2;
	Count[1]  = N - Count[0];
	Offset[0] = 0;
	Offset[1] = Count[0];
	
	for(int d = 0; d < 2; d++)
	{
		GridSize[d].x = (Count[d] - 1)/BlockSize.x + 1; // Enough blocks for this GPU's slice only.
		GridSize[d].y = 1;
		GridSize[d].z = 1;
	}
	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
	// Make sure we actually have two GPUs.
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__, __LINE__);
	if(deviceCount < 2)
	{
		printf("\n This code needs 2 GPUs but only %d found.\n", deviceCount);
		exit(0);
	}
	
	for(int d = 0; d < 2; d++)
	{
		cudaSetDevice(d);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// Full copies of positions and masses on each GPU.
		cudaMalloc(&MGPU[d], N*sizeof(float));
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMalloc(&PGPU[d], N*sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
		
		// Velocity and force only for this GPU's slice.
		cudaMalloc(&VGPU[d], Count[d]*sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMalloc(&FGPU[d], Count[d]*sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	// If the GPUs can talk directly to each other, turn that on. If not, cudaMemcpyPeer
	// still works, it just gets staged through the host.
	int canAccess01, canAccess10;
	cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
	cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
	if(canAccess01 && canAccess10)
	{
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess(1, 0);
		cudaSetDevice(1);
		cudaDeviceEnablePeerAccess(0, 0);
		cudaGetLastError(); // Clear any "already enabled" style error so it doesn't trip our checks.
	}
    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	
	for(int d = 0; d < 2; d++)
	{
		cudaSetDevice(d);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// Everything gets all positions and masses...
		cudaMemcpy(PGPU[d], P, N*sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpy(MGPU[d], M, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// ...but only its own slice of velocities and forces.
		cudaMemcpy(VGPU[d], V + Offset[d], Count[d]*sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpy(FGPU[d], F + Offset[d], Count[d]*sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	cudaSetDevice(0);
}

// n = total number of bodies, offset = first body this GPU owns, count = how many bodies this GPU owns.
// i is the LOCAL index (used for f and v, which only hold this GPU's slice).
// gi is the GLOBAL index (used for p and m, which hold all the bodies).
__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int n, int offset, int count)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < count)
	{
		int gi = i + offset;
		
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(gi != j)
			{
				dx = p[j].x-p[gi].x;
				dy = p[j].y-p[gi].y;
				dz = p[j].z-p[gi].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[gi]*m[j])/(d2) - (h*m[gi]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

// Same local/global index idea as getForces. Only this GPU's slice of p gets updated.
__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int offset, int count)
{	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < count)
	{
		int gi = i + offset;
		
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[gi])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[gi])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[gi])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[gi])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[gi])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[gi])*dt;
		}

		p[gi].x += v[i].x*dt;
		p[gi].y += v[i].y*dt;
		p[gi].z += v[i].z*dt;
	}
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	while(t < RUN_TIME)
	{
		// Launch on both GPUs. Kernel launches are asynchronous, so both GPUs work at the same time.
		for(int d = 0; d < 2; d++)
		{
			cudaSetDevice(d);
			getForces<<<GridSize[d],BlockSize>>>(PGPU[d], VGPU[d], FGPU[d], MGPU[d], G, H, N, Offset[d], Count[d]);
			cudaErrorCheck(__FILE__, __LINE__);
			moveBodies<<<GridSize[d],BlockSize>>>(PGPU[d], VGPU[d], FGPU[d], MGPU[d], Damp, dt, t, Offset[d], Count[d]);
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// Wait for both GPUs to finish moving their bodies before we swap anything.
		for(int d = 0; d < 2; d++)
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// Swap halves: each GPU sends its freshly updated positions to the other GPU.
		cudaMemcpyPeer(PGPU[1] + Offset[0], 1, PGPU[0] + Offset[0], 0, Count[0]*sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyPeer(PGPU[0] + Offset[1], 0, PGPU[1] + Offset[1], 1, Count[1]*sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
		
		if(drawCount == DRAW_RATE) 
		{	
			drawPicture();
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}

int main(int argc, char** argv)
{
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}

