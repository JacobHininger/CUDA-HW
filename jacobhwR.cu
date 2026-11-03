// Name: Jacob Hininger
// N body problem (CPU version)
// nvcc jacobhwR.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

/*
 Explain what you did to fix the code:
 
 1. Added #define NUMBER_OF_SPHERES. Replaced the separate variables for body 1 and body 2
    (px1, px2, vx1, vx2, ...) with arrays indexed by sphere (px[i], vx[i], fx[i], mass[i], ...).
    Changing NUMBER_OF_SPHERES now changes the whole simulation with no other edits.

 2. set_initail_conditions(): each sphere is given a random position and a check is made against
    every sphere already placed so no two start overlapping. This also fixes a bug in the original
    code where dx was computed as px2 - px2 (always 0) instead of px2 - px1. I added a cap on the
    number of placement attempts so the program exits with a message instead of looping forever if
    too many spheres are asked for to fit in the box. Every sphere gets a random velocity,
    a mass, and its own color.

 3.the forces are zeroed first, then a double loop visits every pair (i, j) with j > i
    exactly once. The gravity and collision push-back force is computed for the pair and applied
    to sphere i and the opposite force to sphere j (Newton's third law), summing into fx, fy, fz.
    That makes the cost O(N^2) per step. I also guard against r = 0 to avoid dividing by zero.

 4. both now loop over all the spheres instead of
    repeating the same code for body 1 and body 2.

 5. draw_picture() draws all spheres in a loop, each with its own color.
*/


// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define NUMBER_OF_SPHERES 10
#define MAX_PLACEMENT_TRIES 100000
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 50.0

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

// One entry per sphere
float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES];
float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];
float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
float mass[NUMBER_OF_SPHERES];
float color[NUMBER_OF_SPHERES][3];

// Function prototypes
float randomFloat();
float randomInRange(float, float);
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(double);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

// Random float in [0, 1]
float randomFloat()
{
	return (float)rand()/RAND_MAX;
}

// Random float in [low, high]
float randomInRange(float low, float high)
{
	return low + (high - low)*randomFloat();
}

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	
	float halfRange = (LENGTH_OF_BOX - DIAMETER)/2.0; // Keeps whole sphere inside the box.
	float dx, dy, dz;
	int tries, tooClose;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		tries = 0;
		do
		{
			px[i] = randomInRange(-halfRange, halfRange);
			py[i] = randomInRange(-halfRange, halfRange);
			pz[i] = randomInRange(-halfRange, halfRange);
			
			// Make sure this sphere doesn't overlap any sphere already placed.
			tooClose = 0;
			for(int j = 0; j < i; j++)
			{
				dx = px[i] - px[j];
				dy = py[i] - py[j];
				dz = pz[i] - pz[j];
				if(dx*dx + dy*dy + dz*dz < DIAMETER*DIAMETER)
				{
					tooClose = 1;
					break;
				}
			}
			
			tries++;
			if(tries > MAX_PLACEMENT_TRIES)
			{
				printf("\n Could not fit %d spheres in the box without overlap.", NUMBER_OF_SPHERES);
				printf("\n Reduce NUMBER_OF_SPHERES or DIAMETER, or increase LENGTH_OF_BOX.\n\n");
				exit(0);
			}
		} while(tooClose);
		
		vx[i] = randomInRange(-MAX_VELOCITY, MAX_VELOCITY);
		vy[i] = randomInRange(-MAX_VELOCITY, MAX_VELOCITY);
		vz[i] = randomInRange(-MAX_VELOCITY, MAX_VELOCITY);
		
		mass[i] = 1.0;
		
		// Random bright color so the spheres are easy to tell apart.
		color[i][0] = randomInRange(0.2, 1.0);
		color[i][1] = randomInRange(0.2, 1.0);
		color[i][2] = randomInRange(0.2, 1.0);
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glColor3f(color[i][0], color[i][1], color[i][2]);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(px[i] > halfBoxLength)
		{
			px[i] = 2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		else if(px[i] < -halfBoxLength)
		{
			px[i] = -2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		
		if(py[i] > halfBoxLength)
		{
			py[i] = 2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
		else if(py[i] < -halfBoxLength)
		{
			py[i] = -2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
				
		if(pz[i] > halfBoxLength)
		{
			pz[i] = 2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
		else if(pz[i] < -halfBoxLength)
		{
			pz[i] = -2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
	}
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	
	// Start every step with zero force on every sphere.
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
	
	// Visit each pair of spheres exactly once (j > i).
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for(int j = i + 1; j < NUMBER_OF_SPHERES; j++)
		{
			dx = px[j] - px[i];
			dy = py[j] - py[i];
			dz = pz[j] - pz[i];
						
			r2 = dx*dx + dy*dy + dz*dz;
			if(r2 < 1.0e-12) r2 = 1.0e-12; // Avoid divide by zero if two spheres sit on top of each other.
			r = sqrt(r2);
		
			forceMag =  mass[i]*mass[j]*GRAVITY/r2;
					
			if (r < DIAMETER)
			{
				dvx = vx[j] - vx[i];
				dvy = vy[j] - vy[i];
				dvz = vz[j] - vz[i];
				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}
		
			// Sphere i is pulled toward j, and j is pulled toward i with the opposite force.
			fx[i] += forceMag*dx/r;
			fy[i] += forceMag*dy/r;
			fz[i] += forceMag*dz/r;
			fx[j] -= forceMag*dx/r;
			fy[j] -= forceMag*dy/r;
			fz[j] -= forceMag*dz/r;
		}
	}
}

void move_bodies(double time)
{
	// Leapfrog: the very first step only advances the velocity by half a step.
	float dtFactor = (time == 0.0) ? 0.5*DT : DT;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		vx[i] += dtFactor*(fx[i] - DAMP*vx[i])/mass[i];
		vy[i] += dtFactor*(fy[i] - DAMP*vy[i])/mass[i];
		vz[i] += dtFactor*(fz[i] - DAMP*vz[i])/mass[i];
		
		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	double time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	exit(0);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	printf("\n N-body simulation with %d spheres.", NUMBER_OF_SPHERES);
	printf("\n Change NUMBER_OF_SPHERES in the source to use a different number.");
	printf("\n To stop, hit control c in the window you launched this from.\n\n");
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
