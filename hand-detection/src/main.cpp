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
#include "glcvplane.h"

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"

#include "libfreenect.h"

#include "gstcontrol.h"
#include "glcvcube.h"

#include <math.h>


#define VIDEO_W 640
#define VIDEO_H 480

GLUtils *window = new GLUtils;

GLCVPlane *plane_video1 = new GLCVPlane;

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
uint8_t *rgb_back, *rgb_mid, *rgb_front;

GLuint gl_depth_tex;
GLuint gl_rgb_tex;

freenect_context *f_ctx;
freenect_device *f_dev;
int freenect_led;
int freenect_angle = 0;

freenect_video_format requested_format = FREENECT_VIDEO_YUV_RGB;
freenect_video_format current_format = FREENECT_VIDEO_YUV_RGB;
freenect_resolution requested_resolution = FREENECT_RESOLUTION_MEDIUM;
freenect_resolution current_resolution = FREENECT_RESOLUTION_MEDIUM;

pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;
int got_rgb = 0;
int got_depth = 0;
int depth_on = 1;
int high_value_8bit = 0;

	
IplImage *fore_image;
IplImage *small_fore_image;
IplImage *background;
IplImage *aux;
IplImage *gray;
IplImage *gray_depth;
IplImage *gray_depth_16bits;
IplImage *binary;

IplImage *Hand1;
IplImage *Hand2;


CvSeq *contours;
CvMemStorage *storage;
	
	
void rgb_cb			(freenect_device *dev, void *rgb, uint32_t timestamp);
void depth_cb			(freenect_device *dev, void *v_depth, uint32_t timestamp);
void *freenect_threadfunc	(void *arg);

void Render 	(void);

CvSeq	*ContoursFinder 	(IplImage *edge, 	CvMemStorage *storage);
void	ShowContours 		(IplImage *rgb_img, 	CvSeq *contours);
int 	Calibration 		(CvSeq *contours, CvPoint2D32f center[2], float radius[2], CvBox2D boundingrect[2]);
int 	ConvexHull 		(IplImage *img, CvSeq *contours);
int 	FindHands 		(IplImage *img, CvSeq *contours, CvPoint2D32f center[2], float *depth, float *depthL, float *depthR);

float PhysicalRotation (float force, int direction, int axis);


// key flags
bool flag_save_bg = false;
bool flag_enable_bg_sub = false;
bool flag_only_closer_obj = false;
bool flag_calibration = false;


GLCVCube *cube_video = new GLCVCube;
GSTVideoControl *gst_video1_object = new GSTVideoControl;


// video 1
gchar*	g_pcFrameBuffer1 = NULL;

void *WrapFunction1 (void *obj);
gboolean BusCall1 (GstBus *bus, GstMessage *msg, gpointer data);
void on_handoff1 (GstElement* pFakeSink, GstBuffer* pBuffer, GstPad* pPad);



// Kalman filter
CvKalman* kalman = cvCreateKalman( 2, 1, 0 );
CvMat* state = cvCreateMat( 2, 1, CV_32FC1 ); /* (phi, delta_phi) */
CvMat* process_noise = cvCreateMat( 2, 1, CV_32FC1 );
CvMat* measurement = cvCreateMat( 1, 1, CV_32FC1 );
CvRNG rng = cvRNG(-1);


#define M_FILTER_SIZE 3

CvPoint2D32f m_filter_center_r[M_FILTER_SIZE];
CvPoint2D32f m_filter_center_l[M_FILTER_SIZE];
float m_filter_z[M_FILTER_SIZE]; // left hand (when using only one hand)
float m_filter_zL[M_FILTER_SIZE]; // left hand (when using both hands)
float m_filter_zR[M_FILTER_SIZE]; // right hand (when using both hands)
int ptr_r;	// right  pointer
int ptr_l;	// left pointer
int ptr_z;	// z pointer
int ptr_zL;	// z pointer
int ptr_zR;	// z pointer
CvPoint2D32f MovingAverageFilter_R (CvPoint2D32f value);
CvPoint2D32f MovingAverageFilter_L (CvPoint2D32f value);
float MovingAverageFilter_Z (float value);
float MovingAverageFilter_ZR (float value);
float MovingAverageFilter_ZL (float value);


gboolean BusCall1 (GstBus *bus, GstMessage *msg, gpointer data);
void on_handoff1 (GstElement* pFakeSink, GstBuffer* pBuffer, GstPad* pPad);
void *WrapFunction1 (void *obj);

int DISPLAY_PLANE = 1;
int DISPLAY_CUBE = 1;


int dilate_var = 0;
int erode_var  = 0;
int smooth_var = 0;

#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2

#define CW_DIRECTION 	0
#define CCW_DIRECTION 	1
//--------------------------------------------------------------------------------------
// Name: main()
// Desc: main function (entry point)
//--------------------------------------------------------------------------------------
int main (int argc, char **argv)
{
	char filepath[100];
	
	// Initializate OGLES2
	printf ("\nInitializing OGLES2...");
	window->GLInit ();
	printf ("OK");
	
	printf ("\nCreating the Cube theater...");
	cube_video->CubeCreate(30, 30, 30);
	
	printf ("OK\n");

	printf ("\nInitializing Kinect...");
	if (freenect_init(&f_ctx, NULL) < 0) {
		printf("Failed\n");
		return 1;
	}
	printf ("OK\n");
	
	printf ("\nInitializing GSTControl...");
	gst_init (&argc, &argv);
	gst_video1_object->GSTInit();
	g_pcFrameBuffer1 = (gchar*) malloc (720 * 480 * 2);
	snprintf(filepath, 100, "file:///home/kinect_demos/videos/final_fantasy.avi");
	gst_video1_object->GSTSetURI(filepath);
	// create the pipe line using the mfw_v4lsin, no callback function and the message bus BusCall1
	gst_video1_object->GSTBuildPipeline((char *)"fakesink", (GCallback)on_handoff1, BusCall1);
	// create the thread for this GSTVideoControl (thread used for Buscall and loop)
	if (! gst_video1_object->GSTThreadCreate(WrapFunction1))
		return 0;
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

	rgb_back = (uint8_t*)malloc(freenect_find_video_mode(current_resolution, current_format).bytes);
	rgb_mid = (uint8_t*)malloc(freenect_find_video_mode(current_resolution, current_format).bytes);
	rgb_front = (uint8_t*)malloc(freenect_find_video_mode(current_resolution, current_format).bytes);
	
	window->GenPerspectiveMatrix (0.5, 1.0, -1.0, 1.0, matProj);
	cube_video->CubeSetProjMatrix(matProj);
	
	
	printf ("\nCreating the Theater 1...");
	plane_video1->PlaneCreate(VIDEO_W, VIDEO_H);
	plane_video1->PlaneCreateTex(VIDEO_W, VIDEO_H, 1);
	plane_video1->PlaneSetProjMatrix(matProj);
	gray_depth = cvCreateImage(cvSize (VIDEO_W, VIDEO_H), 8, 1);
	gray_depth_16bits = cvCreateImage(cvSize (VIDEO_W, VIDEO_H), 16, 1);
	fore_image = cvCreateImage (cvSize (640, 480), 8, 3);
	small_fore_image = cvCreateImage (cvSize (320, 240), 8, 3);
	gray = cvCreateImage (cvSize (VIDEO_W, VIDEO_H), 8, 1);
	binary = cvCreateImage (cvSize (VIDEO_W, VIDEO_H), 8, 1);
	
	//cvNamedWindow("main", 0);
	
	
	aux = cvLoadImage("aux.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
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
	char key;
	cvNamedWindow("main", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("main", 360, 240);
	cvShowImage("main", aux);
	
	storage = cvCreateMemStorage(0);
		
	// Change the GSTcontrol to playing state
	gst_video1_object->GSTPlay();

	while (!window->Kbhit ())
	{
		tt = (double)cvGetTickCount();
		Render ();	
		tt = (double)cvGetTickCount() - tt;
		value = tt/(cvGetTickFrequency()*1000.);
		//printf( "\ntime = %gms --- %.2lf FPS", value, 1000.0 / value);
		
		key = cvWaitKey (3);
		
		
		// up arrow key
		if (key == 82) {
			freenect_angle++;
			if (freenect_angle > 30) {
				freenect_angle = 30;
				freenect_set_led(f_dev,LED_YELLOW);
			}
			freenect_set_tilt_degs(f_dev,freenect_angle);
		}
		
		// down arrow key
		if (key == 84) {
			freenect_angle--;
			if (freenect_angle < -30) {
				freenect_angle = -30;
				freenect_set_led(f_dev,LED_YELLOW);
			}
			freenect_set_tilt_degs(f_dev,freenect_angle);
		}

		// left arrow key
		if (key == 81) {
			freenect_set_led(f_dev,LED_GREEN);
		}

		// right arrow key
		if (key == 83) {
			freenect_set_led(f_dev,LED_RED);
		}
		
		
		// segmentation
		if (key == 's') {
			flag_only_closer_obj = !flag_only_closer_obj;
			
			if (flag_only_closer_obj)
			{
				printf ("\nDynamic fore image segmentation enabled (convexhull)...\n");
			}
			else
			{
				printf ("\nDynamic fore image segmentation disabled (convexhull)...\n");
			}


		}

		// bg subtraction
		if (key == 'b') {
			flag_enable_bg_sub = !flag_enable_bg_sub;
			
			if (flag_enable_bg_sub)
			{
				printf ("\nBackground subtraction enabled...\n");
				background = cvLoadImage("background.jpg", CV_LOAD_IMAGE_GRAYSCALE);
			}
			else
			{
				printf ("\nBackground subtraction disabled...\n");
				cvReleaseImage (&background);
			}
		}

		// calibration
		if (key == 'c') {
			flag_calibration = !flag_calibration;
			
			if (flag_calibration)
			{
				printf ("\nDynamic fore image segmentation enabled...\n");
			}
			else
			{
				printf ("\nDynamic fore image segmentation disabled...\n");
			}
		}

	
		// display plane
		if (key == 'o') {
			DISPLAY_PLANE = !DISPLAY_PLANE;
			
			if (DISPLAY_PLANE)
			{
				printf ("\nDisplaying Plane...\n");
			}
			else
			{
				printf ("\nHiding Plane...\n");
			}
		}


		// display cube
		if (key == 'p') {
			DISPLAY_CUBE = !DISPLAY_CUBE;
			
			if (DISPLAY_CUBE)
			{
				printf ("\nDisplaying Cube...\n");
			}
			else
			{
				printf ("\nHiding Cube...\n");
			}
		}

		// change erode value
		if (key == '1') {
			
			if (erode_var > 4) erode_var = 0;
			else erode_var++;
			
			printf ("\nApplying Erode: %d", erode_var);
		}

		// change dilate value
		if (key == '2') {
			
			if (dilate_var > 4) dilate_var = 0;
			else dilate_var++;
			
			printf ("\nApplying Dilate: %d", dilate_var);
		}

		// change smooth value
		if (key == '3') {
			
			if (smooth_var > 5) smooth_var = 0;
			if (! smooth_var) smooth_var++;
			else
			{
				smooth_var+=2;
			}
			
			printf ("\nApplying Low pass filter: %d", smooth_var);
		}

	}
	
	window->GLEnd();
	
	plane_video1->PlaneDestroy();
	
	printf("\nshutting down streams...\n");

	freenect_stop_depth(f_dev);
	//freenect_stop_video(f_dev);

	freenect_close_device(f_dev);
	freenect_shutdown(f_ctx);

	
	cube_video->CubeDestroy();
	
	printf("-- done!\n");
	
	free (rgb_back);
	free (rgb_mid);
	free (rgb_front);
	
	cvReleaseImage(&small_fore_image);
	cvReleaseImage(&fore_image);
	cvReleaseImage(&gray_depth);
	cvReleaseImage(&gray_depth_16bits);
	cvReleaseImage(&background);
	cvReleaseImage(&aux);
	cvReleaseImage(&gray);
	
	cvReleaseMemStorage 	(&storage);	

	return 0;
}




void Render (void)
{
	
	static float r = 0;
	static float g = 0;
	static float b = 0;
	static int cnt = 0;
	
	static int flag_calibration_ok = 0;
	
	static IplImage *result = cvCreateImage (cvSize (VIDEO_W, VIDEO_H), 8, 1);
	
	static int cube_z_value = -150;
	static int cube_x_value = 0;
	static int cube_y_value = 0;
	
	
	static float rotationx = 0;
	static float rotationy = 0;
	static float rotationz = 0;
	
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

	//if (rotationx1 == 360) rotationx1 = 0;
	//else rotationx1 = rotationx1 + 0.3;
	//if (rotationy1 == 360) rotationy1 = 0;
	//else rotationy1 = rotationy1 + 0.7;
	//if (rotationz1 == 360) rotationz1 = 0;
	//else rotationz1 = rotationz1 + 0.5;
	
	/*if (got_rgb) {
		tmp = rgb_front;
		rgb_front = rgb_mid;
		rgb_mid = tmp;
		got_rgb = 0;
	}
	*/

	if (got_depth) {
		got_depth = 0;
		
		cvCopy (gray_depth, result);
	
		if (flag_save_bg)
		{
			cvSaveImage("background.jpg", gray_depth);
		}
		
		
		if (flag_enable_bg_sub)
		{
			cvSub (gray_depth, background, result);

			uchar *ptr_gray;
			int i, j;
		
			for (i = 0; i < gray_depth->height; i++)
			{
				for (j = 0; j < gray_depth->width; j++)
				{
					ptr_gray = cvPtr2D (result, i, j, 0);
					
					if (ptr_gray[0] < 50) 
					{
						ptr_gray[0] = 0;
					}
				}
			}


		}
		
		cvFlip(result, result, 1);
		//cvFlip(gray_depth_16bits, gray_depth_16bits, 1);
		cvCvtColor(result, fore_image, CV_GRAY2RGB);

	}

	
	//plane_video1->PlaneRotate(PLANE_X_AXIS, rotationx1);
	//plane_video1->PlaneRotate(PLANE_Y_AXIS, rotationy1);
	//plane_video1->PlaneRotate(PLANE_Z_AXIS, rotationz1);
	//plane_video1->PlaneSetTexBuf((char *)depth_front, VIDEO_W, VIDEO_H);
	//cvDilate (gray_depth, gray_depth, NULL, 2);
	//cvErode (gray_depth, gray_depth, NULL, 1);
	//cvSmooth(gray_depth, gray_depth, CV_GAUSSIAN, 3, 1);
	//plane_video1->PlaneSetTexBin(result);
	
	
	/*
	uchar *ptr_gray;
	uchar *ptr_rgb;
	int i, j;
	
	for (i = 0; i < gray_depth->height; i++)
	{
		for (j = 0; j < gray_depth->width; j++)
		{
			ptr_gray = cvPtr2D (gray_depth, i, j, 0);
			ptr_rgb = cvPtr2D (fore_image, i, j, 0);
			
			if (ptr_gray[0] >= (high_value_8bit - 20)) 
			{
				ptr_rgb[0] = 255;
				ptr_rgb[1] = 0;
				ptr_rgb[2] = 0;
			}
		}
	}

	cvResize(fore_image, small_fore_image, CV_INTER_LINEAR);
	plane_video1->PlaneSetTex(small_fore_image);
	*/
	
	// calibration
	static CvPoint2D32f center2[2]; 
	static float radius2[2];
	CvBox2D boundingrect[2];

/*	if (flag_calibration)
	{
		cvClearMemStorage(storage);
		contours = ContoursFinder(result, storage);

	
		if (Calibration (contours, center2, radius2, boundingrect))
		{
			printf ("\nCalibrated...\n");
	//		cvCircle (fore_image, cvPoint (cvRound (center2[0].x), cvRound (center2[0].y)), radius2[0], CV_RGB (255, 0, 0), 2, 0);
	//		cvCircle (fore_image, cvPoint (cvRound (center2[1].x), cvRound (center2[1].y)), radius2[1], CV_RGB (0, 0, 255), 2, 0);
	//		cvEllipseBox(fore_image, boundingrect[0], CV_RGB (255, 0, 0), 2, 0);
	//		cvEllipseBox(fore_image, boundingrect[1], CV_RGB (0, 0, 255), 2, 0);
	
			flag_calibration_ok = true;
			flag_calibration = false;
		}
		
		else 
		{
			flag_calibration_ok = false;
		}
	}
	
	else
	{
		
*/		
		if (dilate_var) cvDilate (result, result, NULL, dilate_var);
		if (erode_var) cvErode (result, result, NULL, erode_var);
		if (smooth_var) cvSmooth(result, result, CV_GAUSSIAN, 3, smooth_var);

		cvThreshold (result, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		contours = ContoursFinder(binary, storage);
		static bool flag_hand_detected = false;
		static bool flag_2hands_detected = false;
	  
		if (flag_only_closer_obj)
		{
			int i;
			CvPoint2D32f center_resp[2];
			static CvPoint2D32f center_resp_aux[2];
			static float depth_resp = -200;
			float a; // aux depth var for 1 hand
			float b; // aux depth var for 2 hands (left hand)
			float c; // aux depth var for 2 hands (right hand)
			
			i = FindHands (fore_image, contours, center_resp, &a, &b, &c);
			 
			if (i == 1)
			{
			  
				if (! flag_hand_detected)
				{
					flag_hand_detected = true;
					depth_resp = a;
					center_resp_aux[0].x = center_resp[0].x;
					center_resp_aux[0].y = center_resp[0].y;
					printf ("\nOne Hand Detected: %f\n", a);
				}
				
				// depth motion
				if (a < depth_resp)
				{
					cube_z_value = cube_z_value + 5;
				}
				
				else if (a > depth_resp)
				{
					cube_z_value = cube_z_value - 5;
				}
				
				depth_resp = a;
				
				
				
				// X and Y motion
				if (center_resp[0].x > center_resp_aux[0].x)
				{
					center_resp_aux[0].x  = center_resp[0].x;
					cube_x_value = cube_x_value + 2;
				}

				if (center_resp[0].x < center_resp_aux[0].x)
				{
					center_resp_aux[0].x  = center_resp[0].x;
					cube_x_value = cube_x_value - 2;
				}
				
				if (center_resp[0].y > center_resp_aux[0].y)
				{
					center_resp_aux[0].y  = center_resp[0].y;
					cube_y_value = cube_y_value - 2;
				}

				if (center_resp[0].y < center_resp_aux[0].y)
				{
					center_resp_aux[0].y  = center_resp[0].y;
					cube_y_value = cube_y_value + 2;
				}

			}
			
			else if (i == 2)
			{
				// depth
				//if (c < (b - 10))
				//{
				//	cube_z_value = cube_z_value + 5;
				//}
				
				//if (c > (b + 10))
				//{
				//	cube_z_value = cube_z_value - 5;
				//}
				
				//y axis rotation
				if (! flag_2hands_detected)
				{
					flag_2hands_detected = true;
					depth_resp = a;
					center_resp_aux[1].x = center_resp[1].x;
					center_resp_aux[1].y = center_resp[1].y;
					
					printf ("\nTwo Hands Detected: %f  -  %f\n", b, c);

				}
				
				if (center_resp[1].x > (center_resp_aux[1].x + 5))
				{
					rotationy+=10;
				}
				if (center_resp[1].x < (center_resp_aux[1].x - 5))
				{
					rotationy-=10;
				}
				
				center_resp_aux[1].x = center_resp[1].x;

				if (center_resp[1].y > (center_resp_aux[1].y + 5))
				{
					rotationx+=10;
				}
				if (center_resp[1].y < (center_resp_aux[1].y - 5))
				{
					rotationx-=10;
				}
				
				center_resp_aux[1].y = center_resp[1].y;


			}
			
			else
			{
			
				//printf ("\nNO Hands Detected\n");
				flag_hand_detected = false;

			}
			
		}
		
		else
		{
			if (DISPLAY_PLANE)
			  ShowContours(fore_image, contours);
		}
//	}
	
//	if (flag_calibration_ok)
//	{
		//cvCircle (fore_image, cvPoint (cvRound (center2[0].x), cvRound (center2[0].y)), radius2[0], CV_RGB (255, 0, 0), 2, 0);
		//cvCircle (fore_image, cvPoint (cvRound (center2[1].x), cvRound (center2[1].y)), radius2[1], CV_RGB (0, 0, 255), 2, 0);
		//cvEllipseBox(fore_image, boundingrect[0], CV_RGB (255, 0, 0), 2, 0);
		//cvEllipseBox(fore_image, boundingrect[1], CV_RGB (0, 0, 255), 2, 0);
		
//	}


	if (DISPLAY_PLANE)	
	{
		//plane_video1->PlaneMove(PLANE_X_AXIS, -350);
		//plane_video1->PlaneMove(PLANE_Y_AXIS, -700);
		plane_video1->PlaneMove(PLANE_Z_AXIS, -1500); // uncomment this to get the plane being displayed
		plane_video1->PlaneSetTex(fore_image);
		plane_video1->PlaneDraw();
	}
	
	
	if (DISPLAY_CUBE)
	{
		cube_video->CubeMove(CUBE_X_AXIS, cube_x_value);
		cube_video->CubeMove(CUBE_Y_AXIS, cube_y_value);
		cube_video->CubeMove(CUBE_Z_AXIS, cube_z_value);
	
		cube_video->CubeRotate(CUBE_Y_AXIS, rotationy);
		cube_video->CubeRotate(CUBE_X_AXIS, rotationx);
	
		cube_video->CubeSetFaceTexBuf(FRONT_FACE, g_pcFrameBuffer1, 720, 480);
		cube_video->CubeDraw();
	}
	
	// Swap Buffers.
	// Brings to the native display the current render surface.
	eglSwapBuffers (window->egldisplay, window->eglsurface);
	assert (eglGetError () == EGL_SUCCESS);
	
	cvClearMemStorage(storage);
	
	return;
}

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
	int i;
	
	uint16_t *depth = (uint16_t*)v_depth;

	pthread_mutex_lock(&video_mutex);
	int high_value = 0;
	
	cvSetZero (gray_depth);
	
	if (flag_calibration)
	{
		for (i = 0; i < 640 * 480; i++)
		{
			int x = 2048 - t_gamma[depth[i]];
			if (x > high_value) high_value = x;
			
			//gray_depth_16bits->imageData[i] = x;
		}
		
		//printf ("\nHigh_value: %d", high_value);
		high_value = high_value - 50;
		
		for (i = 0; i < 640 * 480; i++)
		{
			int x = 2048-t_gamma[depth[i]];
			if (x >= high_value)
			{	
				int y = 255-ceil ((0.124 * t_gamma[depth[i]]) + 0.876);
				gray_depth->imageData[i] = y;
//				if (y > high_value_8bit) high_value_8bit = y;
			}
			else
			{
			//	int y = 255-ceil ((0.124 * t_gamma[depth[i]]) + 0.876);
			//	gray_depth->imageData[i] = y;
				gray_depth->imageData[i] =  0;
			}
			
		}
	}
	
	else if (flag_only_closer_obj)
	{
		for (i = 0; i < 640 * 480; i++)
		{
			int x = 2048 - t_gamma[depth[i]];
			if (x > high_value) high_value = x;

			//int y = 255-ceil ((0.124 * t_gamma[depth[i]]) + 0.876);
			//gray_depth_16bits->imageData[i] = x;

		}
		
		//printf ("\nHigh_value: %d", high_value);
		high_value = high_value - 200;
		
		for (i = 0; i < 640 * 480; i++)
		{
			int y = 255-ceil ((0.124 * t_gamma[depth[i]]) + 0.876);
			int x = 2048-t_gamma[depth[i]];
			if (x >= high_value)
			{	
				gray_depth->imageData[i] = y;
//				if (y > high_value_8bit) high_value_8bit = y;
			}
			else
			{
			//	int y = 255-ceil ((0.124 * t_gamma[depth[i]]) + 0.876);
			//	gray_depth->imageData[i] = y;
				gray_depth->imageData[i] =  0 ;
			}
			
		}
	}
	
	else
	{
		for (i = 0; i < 640 * 480; i++)
		{
			int y = 255-ceil ((0.124 * t_gamma[depth[i]]) + 0.876);
			gray_depth->imageData[i] = y;
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
	freenect_set_tilt_degs(f_dev,freenect_angle);
	freenect_set_led(f_dev,LED_RED);
	freenect_set_depth_callback(f_dev, depth_cb);
	freenect_set_video_callback(f_dev, video_cb);
	freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, current_format));
	freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
	freenect_set_video_buffer(f_dev, rgb_back);
	
	freenect_start_depth(f_dev);
	//freenect_start_video(f_dev);

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

// finds hull points
CvSeq *ContoursFinder (IplImage *edge, 	CvMemStorage *storage)
{
	CvSeq* contoursss = NULL;
		
	cvFindContours (edge, storage, &contoursss, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	
	return contoursss;
}

void ShowContours (IplImage *rgb_img, CvSeq *contours)
{
	int i = 0;
	CvPoint2D32f center; 
	float radius;
	CvSeq *aux = contours;

	
	while (aux)
	{
		CvSeq* result = cvApproxPoly (aux, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(aux)*0.02, 0);		
		double a = fabs(cvContourArea(result,CV_WHOLE_SEQ));
		

		if (a > 1000)
		{
			cvMinEnclosingCircle (aux, &center, &radius);
			
			
			printf ("\nCONTOUR %d - AREA: %f",i++, a);
			if (i == 0)
			{
				cvDrawContours (rgb_img, aux, CV_RGB (255, 0, 0), CV_RGB (0, 255, 0), 1, 1, CV_AA, cvPoint(0,0));
				cvCircle (rgb_img, cvPoint (cvRound (center.x), cvRound (center.y)), radius, CV_RGB (255, 0, 0), 2, 0);
			}
				
			else if (i == 1)
			{
				cvDrawContours (rgb_img, aux, CV_RGB (255, 255, 0), CV_RGB (0, 255, 0), 1, 1, CV_AA, cvPoint(0,0));
				cvCircle (rgb_img, cvPoint (cvRound (center.x), cvRound (center.y)), radius, CV_RGB (255, 255, 0), 2, 0);
			}
			
			else
			{
				cvDrawContours (rgb_img, aux, CV_RGB (0, 0, 255), CV_RGB (0, 255, 0), 1, 1, CV_AA, cvPoint(0,0));
				cvCircle (rgb_img, cvPoint (cvRound (center.x), cvRound (center.y)), radius, CV_RGB (0, 0, 255), 2, 0);
			}
		}
		
	
		aux = aux->h_next;
	}

	return;
}

// calibration using 2 hands
int Calibration (CvSeq *contours, CvPoint2D32f center[2], float radius[2], CvBox2D boundingrect[2])
{
	static int x = 0;
	
	CvSeq *aux = contours;
	int i = 0;
	
	freenect_set_led(f_dev,LED_BLINK_RED_YELLOW);
		
	while (aux)
	{ 
		CvSeq* result = cvApproxPoly (aux, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(aux)*0.02, 0);		
		double a = fabs(cvContourArea(result,CV_WHOLE_SEQ));
		
		if (a > 1000)
		{
			cvMinEnclosingCircle (aux, &center[i], &radius[i]);
			boundingrect[i] = cvMinAreaRect2(aux, storage);
			i++;
			if (i > 2) return 0;
		}
		aux = aux->h_next;

	}
	
	if (i != 2)
	{
		x = 0;
		return 0;
	}
	
	x++;

	if (center[0].x < center[1].x)
	{
		CvPoint2D32f aux_center;
		float aux_radius;
		CvBox2D aux_box;
	

		aux_center.x = center[1].x;
		center[1].x = center[0].x;
		center[0].x = aux_center.x;
		
		aux_center.y = center[1].y;
		center[1].y = center[0].y;
		center[0].y = aux_center.y;
		
		aux_radius = radius[1];
		radius[1] =radius[0];
		radius[0] = aux_radius;
		
		aux_box.angle = boundingrect[0].angle;
		boundingrect[1].angle = boundingrect[0].angle;
		boundingrect[0].angle = aux_box.angle;
		
		aux_box.center = boundingrect[0].center;
		boundingrect[1].center = boundingrect[0].center;
		boundingrect[0].center = aux_box.center;

		aux_box.size = boundingrect[0].size;
		boundingrect[1].size = boundingrect[0].size;
		boundingrect[0].size = aux_box.size;
	}
	
	if (x == 10) 
	{
		freenect_set_led(f_dev,LED_GREEN);
		x = 0;
		
		return 1;
	}
	
	else
		return 0;
}

int ConvexHull (IplImage *img, CvSeq *contours)
{
	double result1 = 0;
	double result2 = 0;
	
	CvSeq *hull = NULL;
	CvSeq *defect = NULL; 

	CvMemStorage *dftStorage;
	CvMemStorage *minStorage;
	
	CvRect rect;
	int checkcxt;
	CvBox2D box;
	
	int hulltotal;
	int defecttotal;
	

	//IplImage *rgb_img = cvCreateImage(cvGetSize (img), 8, 3);
	
	minStorage = cvCreateMemStorage();
	dftStorage = cvCreateMemStorage();
	
	checkcxt = cvCheckContourConvexity(contours);
	hull = cvConvexHull2(contours, 0, CV_CLOCKWISE, 0);
	defect = cvConvexityDefects(contours, hull, dftStorage);
        //box = cvMinAreaRect2(contours, minStorage);
		
	//cvCircle (img, cvPoint(box.center.x, box.center.y), 3, CV_RGB(200,0,200), 2, 8, 0);
	//cvEllipse(img, cvPoint(box.center.x, box.center.y), cvSize(box.size.height /2, box.size.width/2), box.angle, 0, 360, CV_RGB(220, 0, 220), 1, 8, 0);
		
	rect = cvBoundingRect(contours, 0);
	//cvRectangle (img, cvPoint(rect.x, rect.y +rect.height), cvPoint(rect.x + rect.width, rect.y), CV_RGB(0,0,255), 2, 8, 0);
	
	//cvSetImageROI (img, rect);
	//Resize (img, smallimg, CV_INTER_LINEAR);
	//cvResetImageROI(img);
	//cvShowImage ("4", smallimg);
	
	//cvZero (rgb_img);
//	cvDrawContours (rgb_img, contours, CV_RGB(255,0,0), CV_RGB(0,255,0), -1, 1, CV_AA, cvPoint(0,0) );

	
	hulltotal = hull->total;
	defecttotal = defect->total; 


	// find the hand face
	//CvPoint2D32f center;
/*	CvPoint center;
	CvSeq *palmseq = NULL;
	int y = 0;
	CvPoint a;

	palmseq = cvCreateSeq (CV_SEQ_ELTYPE_POINT, sizeof(CvContour), sizeof(CvPoint), storageseq);
		 

	for( int i = 0; i < defecttotal - 1; i++) 
	{
		CvConvexityDefect* d = (CvConvexityDefect*) cvGetSeqElem(defect,i);
		
		if(d->depth > 20)
		{
			
			y++;
			a.x = d->depth_point->x;
			a.y = d->depth_point->y;
			cvSeqPush(palmseq, &a);
		}
	}
	
	if (! y)
	{
		cvReleaseImage (&smallimg);
		cvReleaseMat (&map_matrix);
		cvReleaseMemStorage(&storageseq);
		
		return false;		
	}


	CvBox2D palmcircle = cvMinAreaRect2(palmseq, 0);
	//palm is the sequence of yellow dots

	
	center.x = cvRound(palmcircle.center.x);
	center.y = cvRound(palmcircle.center.y);
	//transform float to int

	cvEllipseBox (img, palmcircle, CV_RGB(0,0,255),2,CV_AA,0);
	//Draw palm circle

	cvCircle(img, center, 10,CV_RGB(0, 0, 255),-1,8,0);
	//Draw palm center

	cvClearSeq(palmseq);
*/
	int j = 0;	
	int k = 0;
	for( int i = 0; i < defecttotal; i++) 
	{
		
		CvConvexityDefect* d = (CvConvexityDefect*) cvGetSeqElem(defect,i);
		
	
		if(d->depth > 8)// && (d->depth < 10))
		{		
			
			CvPoint start;
			CvPoint end;
			CvPoint depth;
			

			start.x = d->start->x;
			start.y = d->start->y; //By these 2 statements, we get the green dot.
			

			end.x = d->end->x;
			end.y = d->end->y; //By these 2 statements, we get the white dot.


			depth.x = d->depth_point->x;
			depth.y = d->depth_point->y; //By these 2 statements, we get the yellow dot.

			
			
			//cvLine(img, end, depth, CV_RGB(0,0,255), 2, 8, 0);
			//cvLine(img, depth, start, CV_RGB(0,0,255), 2, 8, 0);
			
			//cvLine(img, center, end, CV_RGB(0,0,255), 2, 8, 0);

			if (depth.x && depth.y);
				//cvCircle (rgb_img, depth, 4, CV_RGB(255,255,0), -1, 8, 0);

			if (start.x && start.y);
				//cvCircle (rgb_img, start, 4, CV_RGB(0,255,0), -1, 8, 0);
				k++; //fingers
			
			if (end.x && end.y)
				//cvCircle (img, end, 8, CV_RGB(255,255,255), -1, 8, 0);

			//char a[10]; 
			//sprintf (a, "%d", j);
			//cvPutText (img, a, p, &axisfont, CV_RGB (0,0,255));

			//printf ("\nDepth: %f, %d", d->depth, j);

			
			j++;
			

		}
		
	}

	//cvCopy (rgb_img, img);
	//if (contours)
	//	contours = cvApproxPoly( contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1 );
	

//	cvDrawContours (aux, contours, cvScalarAll(255), cvScalarAll(255), 100);
	
	//for( ; contours != 0; contours = contours->h_next )
	//{
	//}

//	cvShowImage ("5", rgb_img);
	
	cvReleaseMemStorage(&minStorage);
	cvReleaseMemStorage(&dftStorage);
	//cvReleaseImage (&rgb_img);
	
	return k;
}

int FindHands (IplImage *img, CvSeq *contours, CvPoint2D32f center[2], float *depth, float *depthL, float *depthR)
{
	float radius[2];
	CvBox2D boundingrect[2];
	CvSeq *aux_contours[2];
	
  
	
	CvSeq *aux = contours;
	int i = 0;
	int b = 0;
	
	freenect_set_led(f_dev,LED_BLINK_RED_YELLOW);
		
	while (aux)
	{ 
		CvSeq* result = cvApproxPoly (aux, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(aux)*0.02, 0);		
		double a = fabs(cvContourArea(result,CV_WHOLE_SEQ));
		
		if (a > 1000)
		{
			if (i >= 2) return 0;
			b++;
			cvMinEnclosingCircle (aux, &center[i], &radius[i]);
			boundingrect[i] = cvMinAreaRect2(aux, storage);
			aux_contours[i] = aux;
			i++;

		}
		aux = aux->h_next;

	}
	
	if (b == 1)
	{
		center[0] = MovingAverageFilter_R (center[0]);
		  
		int x1 = cvRound (center[0].x);
		int y1 = cvRound (center[0].y);
		uchar *depth_value1;
				  
		depth_value1 = cvPtr2D (img, y1, x1, 0);
		
		(*depth) = MovingAverageFilter_Z(depth_value1[0]);
		//(*depth) = depth_value1[0];
		

		//printf ("\nCENTER 1 : (%d, %d) - %d", x1, y1, depth_value1[0]);
		  
		int hull_res = ConvexHull (img, aux_contours[0]);
		  
		if ((hull_res > 4))// && (hull_res > 6))
		{
			if (DISPLAY_PLANE)
				cvCircle (img, cvPoint (x1, y1), sqrt (pow ((*depth) - 200, 2)), CV_RGB (255, 0, 0), 5, 0);
			
			//printf ("\nHull: %d", hull_res);
			return 1;
		}
		  
		else
			return 0;
	}
	
	if (b == 2)
	{
	
		if (center[0].x > center[1].x)
		{
			CvPoint2D32f aux_center;
			float aux_radius;
			CvBox2D aux_box;
			CvSeq *aux_cont;
			
		

			aux_center.x = center[1].x;
			center[1].x = center[0].x;
			center[0].x = aux_center.x;
			
			aux_cont = aux_contours[1];
			aux_contours[1] = aux_contours[0];
			aux_contours[0] = aux_cont;
			
			aux_center.y = center[1].y;
			center[1].y = center[0].y;
			center[0].y = aux_center.y;
			
			aux_radius = radius[1];
			radius[1] =radius[0];
			radius[0] = aux_radius;
			
			aux_box.angle = boundingrect[0].angle;
			boundingrect[1].angle = boundingrect[0].angle;
			boundingrect[0].angle = aux_box.angle;
			
			aux_box.center = boundingrect[0].center;
			boundingrect[1].center = boundingrect[0].center;
			boundingrect[0].center = aux_box.center;

			aux_box.size = boundingrect[0].size;
			boundingrect[1].size = boundingrect[0].size;
			boundingrect[0].size = aux_box.size;
		}
		
		center[0] = MovingAverageFilter_R (center[0]);
		center[1] = MovingAverageFilter_L (center[1]);
		
		int x1 = cvRound (center[0].x);
		int y1 = cvRound (center[0].y);
		int x2 = cvRound (center[1].x);
		int y2 = cvRound (center[1].y);
		
		uchar *depth_value1;
		uchar *depth_value2;
		
		depth_value1 = cvPtr2D (img, y1, x1, 0);
		depth_value2 = cvPtr2D (img, y2, x2, 0);
		
		(*depthL) = MovingAverageFilter_ZL(depth_value1[0]);
		(*depthR) = MovingAverageFilter_ZR(depth_value2[0]);

//		printf ("\nCENTER 1 : (%d, %d) - %d", x1, y1, depth_value1[0]);
//		printf ("\nCENTER 2 : (%d, %d)- %d", x2, y2, depth_value2[0]); 
		
		//cvCircle (img, cvPoint (x1, y1), sqrt (pow (depth_value1[0] - 200, 2)), CV_RGB (255, 0, 0), 5, 0);



		//cvEllipseBox(img, boundingrect[0], CV_RGB (255, 0, 0), 2, 0);
		//cvEllipseBox(img, boundingrect[1], CV_RGB (0, 0, 255), 2, 0);
		
		int hull_res1 = ConvexHull (img, aux_contours[0]);
		  
		if ((hull_res1 > 4))// && (hull_res1 > 6))
		{
			if (DISPLAY_PLANE)
			{
				cvCircle (img, cvPoint (x1, y1), sqrt (pow ((*depthL) - 200, 2)), CV_RGB (255, 0, 0), 5, 0);
				cvCircle (img, cvPoint (x2, y2), sqrt (pow ((*depthR) - 200, 2)), CV_RGB (0, 0, 255), 5, 0);
			}
			return 2;
		}
		  
		else
			return 0;

	}
	
	else
	
		return 0;
}

CvPoint2D32f MovingAverageFilter_R (CvPoint2D32f value)
{
	CvPoint2D32f average;
	float sum_x = 0;
	float sum_y = 0;
	int i;

	if (ptr_r >= M_FILTER_SIZE)
	{
		ptr_r = 0;
	}			

	m_filter_center_r[ptr_r].x = value.x;
	m_filter_center_r[ptr_r].y = value.y;

	for (i = 0; i < M_FILTER_SIZE; i++)
	{
		sum_x = sum_x + m_filter_center_r[i].x;
		sum_y = sum_y + m_filter_center_r[i].y;
	}
	
	average.x = sum_x / i;
	average.y = sum_y / i;

	ptr_r++;

	return average;	
}

CvPoint2D32f MovingAverageFilter_L (CvPoint2D32f value)
{
	CvPoint2D32f average;
	float sum_x = 0;
	float sum_y = 0;
	int i;

	if (ptr_l >= M_FILTER_SIZE)
	{
		ptr_l = 0;
	}			

	m_filter_center_l[ptr_l].x = value.x;
	m_filter_center_l[ptr_l].y = value.y;

	for (i = 0; i < M_FILTER_SIZE; i++)
	{
		sum_x = sum_x + m_filter_center_l[i].x;
		sum_y = sum_y + m_filter_center_l[i].y;
	}
	
	average.x = sum_x / i;
	average.y = sum_y / i;

	ptr_l++;

	return average;	
}

float MovingAverageFilter_Z (float value)
{
	float average;
	float sum = 0;
	int i;

	if (ptr_z >= M_FILTER_SIZE)
	{
		ptr_z = 0;
	}			

	m_filter_z[ptr_z] = value;
	
	for (i = 0; i < M_FILTER_SIZE; i++)
	{
		sum = sum + m_filter_z[i];
	}
	
	average = sum / i;

	ptr_z++;

	return average;	
}

float MovingAverageFilter_ZR (float value)
{
	float average;
	float sum = 0;
	int i;

	if (ptr_zR >= M_FILTER_SIZE)
	{
		ptr_zR = 0;
	}			

	m_filter_zR[ptr_zR] = value;
	
	for (i = 0; i < M_FILTER_SIZE; i++)
	{
		sum = sum + m_filter_zR[i];
	}
	
	average = sum / i;

	ptr_zR++;

	return average;	
}
float MovingAverageFilter_ZL (float value)
{
	float average;
	float sum = 0;
	int i;

	if (ptr_zL >= M_FILTER_SIZE)
	{
		ptr_zL = 0;
	}			

	m_filter_zL[ptr_zL] = value;
	
	for (i = 0; i < M_FILTER_SIZE; i++)
	{
		sum = sum + m_filter_zL[i];
	}
	
	average = sum / i;

	ptr_zL++;

	return average;	
}
// VIDEO 1
// message bus for the current GSTVideoControl object, must have replicated for each created object
gboolean BusCall1 (GstBus *bus, GstMessage *msg, gpointer data)
{
  	switch (GST_MESSAGE_TYPE (msg))
	{
		case GST_MESSAGE_EOS:	g_print ("End of stream\n");
					gst_video1_object->GSTSeekAbsolute(0);
					break;

    		case GST_MESSAGE_ERROR:{
						gchar  *debug;
      						GError *error;
						gst_message_parse_error (msg, &error, &debug);
      						g_free (debug);
	
      						g_printerr ("Error: %s\n", error->message);
      						g_error_free (error);
	
					}
		      		
    		default:		break;
  	}

  return TRUE;

}

// just to create the pthread inside the GSTVideoControl class, this wrap function must be replicate for every object
void *WrapFunction1 (void *obj)
{
	GSTVideoControl *aux = reinterpret_cast <GSTVideoControl *> (obj);
	aux->GSTLoopFunction();
	
	return NULL;

}

//handoff function, called every frame
void on_handoff1 (GstElement* pFakeSink, GstBuffer* pBuffer, GstPad* pPad)
{
	int video_w;
	int video_h;
	
	video_w = gst_video1_object->GSTGetPadWidth (pPad);
	video_h = gst_video1_object->GSTGetPadHeight (pPad);

	gst_buffer_ref (pBuffer);
	memmove (g_pcFrameBuffer1, GST_BUFFER_DATA (pBuffer), video_w * video_h * 2);
	gst_buffer_unref (pBuffer);
}



// the function below is not being used, yet !
float PhysicalRotation (float force, int direction, int axis)
{
    float resp;
    
    float u = 0.4;
    float friction;
    
    static float old_force = force;
    
    friction = u * force;
    force = old_force - friction;
    old_force = force;
       
    
    
    if (axis == X_AXIS)
    {
      
    }
    
    if (axis == Y_AXIS)
    {
    }

    if (axis == Z_AXIS)
    {
    }
    
    
    return resp;
  
}