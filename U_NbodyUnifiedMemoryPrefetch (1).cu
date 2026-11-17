// Name:
// nBody run on all available GPUs. 
// nvcc U_NbodyUnifiedMemoryPrefetch.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 It automatically detects the number of available GPUs on the machine and runs using all of them.
 Rewrite the code using CUDA unified memory to make it simpler and use cudaMemPrefetchAsyn to make
 the memory movement faster.
*/

/*
 Purpose:
 To see how Nbody code is run on many GPUs and learn how to simplify it with unified memory then speed it up 
 with prefectched memory.
*/

/*
 Explain what you did to fix the code:
 1. Removed the per-GPU arrays (PGPU, VGPU, FGPU, MGPU) and the cudaMalloc/cudaMemcpy calls. P, V, F, and M are now
    single cudaMallocManaged arrays that the CPU and every GPU can use through the same pointer.
 2. Removed the GPU-to-GPU cudaMemcpyAsync loop. With one shared P, each GPU writes its own slice and the others see it.
 3. Added cudaMemPrefetchAsync so pages are moved in bulk before the kernels need them instead of one page fault at a time:
      - V, F, M: each GPU only touches its own slice, so each slice is prefetched to its GPU once, after setup.
      - P: every GPU reads all of it every step, so it is marked cudaMemAdviseSetReadMostly (each GPU keeps a read copy)
        and prefetched to every GPU at the start of each step.
 4. Split the step into two phases (all GPUs getForces -> sync -> all GPUs moveBodies -> sync). Before, each GPU had its
    own copy of P so moveBodies could follow getForces right away. Now P is shared, so moveBodies on one GPU would change
    positions while another GPU was still reading them in getForces.
 5. Added a (local < nPerGPU) check to the kernels. GridSize is rounded up to whole blocks, so the extra threads of one GPU
    would land in the next GPU's slice and update those bodies twice. That was harmless with separate copies but is a bug
    with shared memory.
 6. Fixed NPerGPU = ceil(N/GPUs). The old formula (N + N%GPUs)/GPUs can round down (e.g. N=1001 on 4 GPUs gives 250*4 = 1000 < N).
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
#define RUN_TIME 1.0

// Globals
int N;
int NPerGPU; // Amount of vector on each GPU.
int NumberOfGpus;
int NN; // N rounded up so it divides evenly across the GPUs (NumberOfGpus*NPerGPU).
float3 *P, *V, *F; // Unified (managed) memory, usable from the CPU and all GPUs.
float *M; 
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void prefetch(void *, size_t, int);
void adviseReadMostly(void *, size_t);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int);
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

// Wrappers so this compiles on CUDA 12 and older as well as CUDA 13+, where the prefetch/advise API takes a cudaMemLocation.
void prefetch(void *ptr, size_t bytes, int device)
{
#if CUDART_VERSION >= 13000
	cudaMemLocation loc;
	loc.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
	loc.id = (device == cudaCpuDeviceId) ? 0 : device;
	cudaMemPrefetchAsync(ptr, bytes, loc, 0, 0);
#else
	cudaMemPrefetchAsync(ptr, bytes, device, 0);
#endif
}

void adviseReadMostly(void *ptr, size_t bytes)
{
#if CUDART_VERSION >= 13000
	cudaMemLocation loc;
	loc.type = cudaMemLocationTypeDevice;
	loc.id = 0;
	cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, loc);
#else
	cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, 0);
#endif
}

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// P is unified memory so the CPU can read it directly. Prefetching it to the CPU first moves it in one go
	// instead of faulting page by page. nBody() has already synced all GPUs.
	cudaSetDevice(0);
	prefetch(P, N*sizeof(float3), cudaCpuDeviceId);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	
	for(int i=0; i<N; i++)
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
	
	N = 1001;
	
	cudaGetDeviceCount(&NumberOfGpus);
	if(NumberOfGpus == 0)
	{
		printf("\n Dude, you don't even have a GPU. Sorry, you can't play with us. Call NVIDIA and buy a GPU — loser!\n");
		exit(0);
	}
	else
	{
		printf("\n You will be running on %d GPU(s)\n", NumberOfGpus);
	}
	
	// Rounding up so NumberOfGpus*NPerGPU is always at least N.
	NPerGPU = (N + NumberOfGpus - 1)/NumberOfGpus;
		
	BlockSize.x = 128;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (NPerGPU - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
	
    	Damp = 0.5;
    	
	// !! Important: Making the arrays a little bigger if N does not divide evenly so each GPU's slice (and its prefetch)
	// stays inside memory we own.
	NN = NumberOfGpus*NPerGPU;
	
	// Unified memory: one pointer each, visible to the CPU and every GPU.
	cudaMallocManaged(&M, NN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMallocManaged(&P, NN*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMallocManaged(&V, NN*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMallocManaged(&F, NN*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the global sphere and setting the initial velocity, initial force, and mass.
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
			
			// Making sure the body centers are at least a diameter apart.
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
	
	// P is read in full by every GPU, so let each GPU keep a read copy of it.
	adviseReadMostly(P, NN*sizeof(float3));
	
	// Zeroing any padding past N so nothing uninitialized sits in the arrays.
	for(int i = N; i < NN; i++)
	{
		P[i].x = P[i].y = P[i].z = 0.0f;
		V[i].x = V[i].y = V[i].z = 0.0f;
		F[i].x = F[i].y = F[i].z = 0.0f;
		M[i] = 1.0f;
	}
	
	// Each GPU only works on its own slice of V, F, and M, so move each slice to its GPU now.
	// (P is prefetched to every GPU at the start of each step in nBody().)
	for(int i = 0; i < NumberOfGpus; i++)
	{
		cudaSetDevice(i);
		prefetch(&V[i*NPerGPU], NPerGPU*sizeof(float3), i);
		prefetch(&F[i*NPerGPU], NPerGPU*sizeof(float3), i);
		prefetch(&M[i*NPerGPU], NPerGPU*sizeof(float),  i);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	for(int i = 0; i < NumberOfGpus; i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
	}
		
	printf("\n Setup finished.\n");
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int nPerGPU, int n, int device)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int offset = nPerGPU*device;
	int local = threadIdx.x + blockDim.x*blockIdx.x;
	int i = local + offset;
	
	// local < nPerGPU keeps this GPU out of the next GPU's slice (the grid is rounded up to whole blocks).
	if(local < nPerGPU && i < n)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int nPerGPU, int n, int device)
{
	int offset = nPerGPU*device;	
	int local = threadIdx.x + blockDim.x*blockIdx.x;
	int i = local + offset;
	
	if(local < nPerGPU && i < n)
	{
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;
	
	printf("\n Simulation is running with %d bodies.\n", N);
	
	while(t < RUN_TIME)
	{
		// Phase 1: every GPU gets the current positions and computes forces for its slice.
		for(int i = 0; i < NumberOfGpus; i++)
		{
			cudaSetDevice(i);
			prefetch(P, NN*sizeof(float3), i); // Read copy of all positions on GPU i.
			getForces<<<GridSize,BlockSize>>>(P, V, F, M, G, H, NPerGPU, N, i);
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// All forces must be done before anyone moves, because moving writes the P that the other GPUs are reading.
		for(int i = 0; i < NumberOfGpus; i++)
		{
			cudaSetDevice(i);
			cudaDeviceSynchronize();
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// Phase 2: every GPU moves its own slice of bodies.
		for(int i = 0; i < NumberOfGpus; i++)
		{
			cudaSetDevice(i);
			moveBodies<<<GridSize,BlockSize>>>(P, V, F, M, Damp, dt, t, NPerGPU, N, i);
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// Syncing CPU with GPUs.
		for(int i = 0; i < NumberOfGpus; i++)
		{
			cudaSetDevice(i);
			cudaDeviceSynchronize();
			cudaErrorCheck(__FILE__, __LINE__);
		}

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
	glutCreateWindow("Nbody");
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

