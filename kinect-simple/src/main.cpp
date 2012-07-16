/************************-***************************************************
 *   Copyright (C) 2011 by Andre L. V. da Silva   									*
 *   andreluizeng@yahoo.com.br   														*
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

//--------------------------------------------------------------------------------------
// File: main.cpp
//--------------------------------------------------------------------------------------
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

#include <termios.h>
#include <unistd.h>
#include <time.h>

#include <pthread.h>

#include "glutils.h"
#include "glplane.h"

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"

#include "libfreenect.h"


#define VIDEO_W 640
#define VIDEO_H 480

GLUtils *window = new GLUtils;

GLPlane *plane_video1 = new GLPlane;
GLPlane *plane_video2 = new GLPlane;
GLPlane *plane_video3 = new GLPlane;

int thread_id;

float matProj[16] = {0};
float matModel[9] = {0};



// kinect
uint16_t t_gamma[2048];
pthread_t freenect_thread;
volatile int die = 0;

int depth_window;
int video_window;

pthread_mutex_t depth_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t video_mutex = PTHREAD_MUTEX_INITIALIZER;

// back: owned by libfreenect (implicit for depth)
// mid: owned by callbacks, "latest frame ready"
// front: owned by GL, "currently being drawn"
uint8_t *depth_mid, *depth_front;
uint8_t *rgb_back, *rgb_mid, *rgb_front;

GLuint gl_depth_tex;
GLuint gl_rgb_tex;

freenect_context *f_ctx;
freenect_device *f_dev;
int freenect_led;

freenect_video_format requested_format = FREENECT_VIDEO_YUV_RGB;
freenect_video_format current_format = FREENECT_VIDEO_YUV_RGB;
freenect_resolution requested_resolution = FREENECT_RESOLUTION_MEDIUM;
freenect_resolution current_resolution = FREENECT_RESOLUTION_MEDIUM;

pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;
int got_rgb = 0;
int got_depth = 0;
int depth_on = 1;


void rgb_cb			(freenect_device *dev, void *rgb, uint32_t timestamp);
void depth_cb			(freenect_device *dev, void *v_depth, uint32_t timestamp);
void *freenect_threadfunc	(void *arg);

void Render 	(void);

//--------------------------------------------------------------------------------------
// Name: main()
// Desc: main function (entry point)
//--------------------------------------------------------------------------------------
int main (int argc, char **argv)
{
	// Initializate OGLES2
	printf ("\nInitializing OGLES2...");
	window->GLInit ();
	printf ("OK");
	
	printf ("\nInitializing Kinect...");
	if (freenect_init(&f_ctx, NULL) < 0) {
		printf("Failed\n");
		return 1;
	}
	printf ("OK\n");

	freenect_set_log_level(f_ctx, FREENECT_LOG_DEBUG);
	freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));
	
	int nr_devices = freenect_num_devices (f_ctx);
	printf ("\t\tNumber of devices found: %d\n", nr_devices);

	int user_device_number = 0;
	if (argc > 1)
		user_device_number = atoi(argv[1]);

	if (nr_devices < 1)
		return 1;

	if (freenect_open_device(f_ctx, &f_dev, user_device_number) < 0) {
		printf("\t\tCould not open device\n");
		return 1;
	}
	
	int i;
	for (i=0; i<2048; i++) {
		float v = i/2048.0;
		v = powf(v, 3)* 6;
		t_gamma[i] = v*6*256;
	}

	depth_mid = (uint8_t*)malloc(640*480*3);
	depth_front = (uint8_t*)malloc(640*480*3);
	rgb_back = (uint8_t*)malloc(freenect_find_video_mode(current_resolution, current_format).bytes);
	rgb_mid = (uint8_t*)malloc(freenect_find_video_mode(current_resolution, current_format).bytes);
	rgb_front = (uint8_t*)malloc(freenect_find_video_mode(current_resolution, current_format).bytes);
	
	window->GenPerspectiveMatrix (0.5, 1.0, -1.0, 1.0, matProj);
	
	printf ("\nCreating the Theater 1...");
	plane_video1->PlaneCreate(VIDEO_W, VIDEO_H);
	plane_video1->PlaneCreateTex(VIDEO_W, VIDEO_H);
	plane_video1->PlaneSetProjMatrix(matProj);
	printf ("OK\n");

	printf ("\nCreating the Theater 2...");
	plane_video2->PlaneCreate(VIDEO_W, VIDEO_H);
	plane_video2->PlaneCreateTex(VIDEO_W, VIDEO_H);
	plane_video2->PlaneSetProjMatrix(matProj);
	printf ("OK\n");
	
	
	// threads for the kinect processes
	int res;
	res = pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
	if (res) {
		printf("pthread_create failed\n");
		return 1;
	}


	// keep the app runing until the end_of_stream is not reached.
	double tt;	
	double value;
	while (!window->Kbhit ())
	{
		tt = (double)cvGetTickCount();
		Render ();	
		tt = (double)cvGetTickCount() - tt;
		value = tt/(cvGetTickFrequency()*1000.);
		printf( "\ntime = %gms --- %.2lf FPS", value, 1000.0 / value);
		
		//cvWaitKey (30);

	}
	
	window->GLEnd();
	
	plane_video1->PlaneDestroy();
	plane_video2->PlaneDestroy();
	
	printf("\nshutting down streams...\n");

	freenect_stop_depth(f_dev);
	freenect_stop_video(f_dev);

	freenect_close_device(f_dev);
	freenect_shutdown(f_ctx);

	printf("-- done!\n");
	
	free (depth_front);
	free (depth_mid);
	free (rgb_back);
	free (rgb_mid);
	free (rgb_front);

	return 0;
}




void Render (void)
{
	
	static float r = 0;
	static float g = 0;
	static float b = 0;
	static int cnt = 0;
	
	
	static float rotationx1 = 0;
	static float rotationy1 = 0;
	static float rotationz1 = 0;
	
	static float rotationx2 = 0;
	static float rotationy2 = 0;
	static float rotationz2 = 0;
	
	// Clear the colorbuffer and depth-buffer -- just playing with some colors
	if (cnt == 0)
	{
		r = 0;
		g = 0;
		
		if (b >= 1.0)
		{
			cnt = 1;
		}
		else
		{
			b = b + 0.01;
			
		}
	}
	else if (cnt == 1)
	{
		r = 0;
		g = 0;
		
		if (b <= 0)
		{
			cnt = 2;
		}
		else
		{
			b = b - 0.01;
		}
	}

	else if (cnt == 2)
	{
		r = 0;
		b = 0;
		
		if (g >= 1.0)
		{
			cnt = 3;
		}
		else
		{
			g = g + 0.01;
		}
	}


	else if (cnt == 3)
	{
		r = 0;
		b = 0;
		
		if (g <= 0)
		{
			cnt = 4;
		}
		else
		{
			g = g - 0.01;
		}
	}
	else if (cnt == 4)
	{
		g = 0;
		b = 0;
		
		if (r >= 1.0)
		{
			cnt = 5;
		}
		else
		{
			r = r + 0.01;
		}
	}


	else if (cnt == 5)
	{
		g = 0;
		b = 0;
		
		if (r <= 0)
		{
			cnt = 6;
		}
		else
		{
			r = r - 0.01;
		}
	}

	
	if (cnt == 6)
	{
		r = 0;
		g = 0;
		
		if (b >= 1.0)
		{
			cnt = 7;
		}
		else
		{
			b = b + 0.01;
			
		}
	}

	if (cnt == 7)
	{
		r = 0;
		
		if (g >= 1.0)
		{
			cnt = 8;
		}
		else
		{
			g = g + 0.01;
			
		}
	}
	
	if (cnt == 8)
	{
		if (r >= 1.0)
		{
			cnt = 9;
		}
		else
		{
			r = r + 0.01;
			
		}
	}
	
	if (cnt == 9)
	{
		if (b <= 0)
		{
			cnt = 10;
		}
		else
		{
			b = b - 0.01;
			
		}
	}

	if (cnt == 10)
	{
		
		if (g <= 0.0)
		{
			cnt = 11;
		}
		else
		{
			g = g - 0.01;
			
		}
	}
	
	if (cnt == 11)
	{
		if (r <= 0.0)
		{
			cnt = 0;
		}
		else
		{
			r = r - 0.01;
			
		}
	}

		
	glClearColor (r, g, b, 1.0f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (rotationx1 == 360) rotationx1 = 0;
	else rotationx1 = rotationx1 + 0.3;
	if (rotationy1 == 360) rotationy1 = 0;
	else rotationy1 = rotationy1 + 0.7;
	if (rotationz1 == 360) rotationz1 = 0;
	else rotationz1 = rotationz1 + 0.5;

	if (rotationx2 <= 0) rotationx2 = 360;
	else rotationx2 = rotationx2 - 0.5;
	if (rotationy2 == 360) rotationy2 = 0;
	else rotationy2 = rotationy2 + 0.5;
	if (rotationz2 <= 0) rotationz2 = 360;
	else rotationz2 = rotationz2 - 0.3;
	
	uint8_t *tmp;
	if (got_depth) {
		tmp = depth_front;
		depth_front = depth_mid;
		depth_mid = tmp;
		got_depth = 0;
		
	}
	if (got_rgb) {
		tmp = rgb_front;
		rgb_front = rgb_mid;
		rgb_mid = tmp;
		got_rgb = 0;
	}
	
	
	plane_video1->PlaneMove(PLANE_X_AXIS, -350);
	//plane_video1->PlaneMove(PLANE_Y_AXIS, -700);
	plane_video1->PlaneMove(PLANE_Z_AXIS, -3000);
	//plane_video1->PlaneRotate(PLANE_X_AXIS, rotationx1);
	//plane_video1->PlaneRotate(PLANE_Y_AXIS, rotationy1);
	//plane_video1->PlaneRotate(PLANE_Z_AXIS, rotationz1);
	plane_video1->PlaneSetTexBuf(depth_front, VIDEO_W, VIDEO_H);
	plane_video1->PlaneDraw();
	
	
	
	plane_video2->PlaneMove(PLANE_X_AXIS, 350);
	//plane_video2->PlaneMove(PLANE_Y_AXIS, -700);
	plane_video2->PlaneMove(PLANE_Z_AXIS, -3000);
	//plane_video2->PlaneRotate(PLANE_X_AXIS, rotationx2);
	//plane_video2->PlaneRotate(PLANE_Y_AXIS, rotationy2);
	//plane_video2->PlaneRotate(PLANE_Z_AXIS, rotationz2);
	plane_video2->PlaneSetTexBuf(rgb_front, VIDEO_W, VIDEO_H);
	plane_video2->PlaneDraw();
		
	
	// Swap Buffers.
	// Brings to the native display the current render surface.
	eglSwapBuffers (window->egldisplay, window->eglsurface);
	assert (eglGetError () == EGL_SUCCESS);
	return;
}


void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
	int i;
	uint16_t *depth = (uint16_t*)v_depth;

	pthread_mutex_lock(&video_mutex);
	
	for (i=0; i<640*480; i++) {
		int pval = t_gamma[depth[i]];
		int lb = pval & 0xff;
		
		//printf ("\ndepth mid: %d - %d - %d", depth[i],  pval, lb);		
		
		switch (pval>>8) {
			case 0:
				depth_mid[3*i+0] = 255;
				depth_mid[3*i+1] = 255-lb;
				depth_mid[3*i+2] = 255-lb;
				break;
			case 1:
				depth_mid[3*i+0] = 255;
				depth_mid[3*i+1] = lb;
				depth_mid[3*i+2] = 0;
				break;
			case 2:
				depth_mid[3*i+0] = 255-lb;
				depth_mid[3*i+1] = 255;
				depth_mid[3*i+2] = 0;
				break;
			case 3:
				depth_mid[3*i+0] = 0;
				depth_mid[3*i+1] = 255;
				depth_mid[3*i+2] = lb;
				break;
			case 4:
				depth_mid[3*i+0] = 0;
				depth_mid[3*i+1] = 255-lb;
				depth_mid[3*i+2] = 255;
				break;
			case 5:
				depth_mid[3*i+0] = 0;
				depth_mid[3*i+1] = 0;
				depth_mid[3*i+2] = 255-lb;
				break;
			default:
				depth_mid[3*i+0] = 0;
				depth_mid[3*i+1] = 0;
				depth_mid[3*i+2] = 0;
				break;
		}
		
	}
	got_depth++;
	pthread_mutex_unlock(&video_mutex);
}


void video_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
	pthread_mutex_lock(&video_mutex);

	// swap buffers
	assert (rgb_back == rgb);
	rgb_back = rgb_mid;
	freenect_set_video_buffer(dev, rgb_back);
	rgb_mid = (uint8_t*)rgb;

	got_rgb++;
	pthread_mutex_unlock(&video_mutex);
}

void *freenect_threadfunc(void *arg)
{
	freenect_set_led(f_dev,LED_RED);
	freenect_set_depth_callback(f_dev, depth_cb);
	freenect_set_video_callback(f_dev, video_cb);
	freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, current_format));
	freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
	freenect_set_video_buffer(f_dev, rgb_back);
	
	freenect_start_depth(f_dev);
	freenect_start_video(f_dev);

	int status = 0;
	while (!die && status >= 0) {
		status = freenect_process_events(f_ctx);
		
		/*if (requested_format != current_format || requested_resolution != current_resolution) {
			freenect_stop_video(f_dev);
			freenect_set_video_mode(f_dev, freenect_find_video_mode(requested_resolution, requested_format));
			pthread_mutex_lock(&video_mutex);
			free(rgb_back);
			free(rgb_mid);
			free(rgb_front);
			rgb_back = (uint8_t*)malloc(freenect_find_video_mode(requested_resolution, requested_format).bytes);
			rgb_mid = (uint8_t*)malloc(freenect_find_video_mode(requested_resolution, requested_format).bytes);
			rgb_front = (uint8_t*)malloc(freenect_find_video_mode(requested_resolution, requested_format).bytes);
			current_format = requested_format;
			current_resolution = requested_resolution;
			pthread_mutex_unlock(&video_mutex);
			freenect_set_video_buffer(f_dev, rgb_back);
			freenect_start_video(f_dev);
		}*/
		
		if (requested_format != current_format) {
			//freenect_stop_video(f_dev);
			//freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, requested_format));
			//freenect_start_video(f_dev);
			current_format = requested_format;
		}

	}

	if (status < 0) {
		printf("Something went terribly wrong.  Aborting.\n");
		return NULL;
	}

	return NULL;
}
