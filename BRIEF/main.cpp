/*
  Copyright 2010 Computer Vision Lab,
  Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
  All rights reserved.

  Authors: Eray Molla, Michael Calonder, and Vincent Lepetit

  This file is part of the BRIEF_demo software.

  BRIEF_demo is  free software; you can redistribute  it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free  Software Foundation; either  version 2 of the  License, or
  (at your option) any later version.

  BRIEF_demo is  distributed in the hope  that it will  be useful, but
  WITHOUT  ANY   WARRANTY;  without  even  the   implied  warranty  of
  MERCHANTABILITY  or FITNESS FOR  A PARTICULAR  PURPOSE. See  the GNU
  General Public License for more details.
  
  You should  have received a copy  of the GNU  General Public License
  along  with   BRIEF_demo;  if  not,  write  to   the  Free  Software
  Foundation,  Inc.,  51  Franklin  Street, Fifth  Floor,  Boston,  MA
  02110-1301, USA
*/

#include "BRIEF.h"

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

/************************************ GLOBAL CONSTANTES ***************************************/

// Frame width and height of the capture
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 480;

// Maximum number of keypoint matches allowed by the program
static const int MAXIMUM_NUMBER_OF_MATCHES = 5000;

// Minimum  scale ratio of  the program.  It indicates  that templates
// having scales  [0.5, 1]  of the original  template will be  used to
// generate new templates. Scales are determined in a logarithm base.
static const float SMALLEST_SCALE_CHANGE = 0.5;

// Number of different scales used to generate the templates.
static const int NUMBER_OF_SCALE_STEPS = 3;

// Number  of   different  rotation   angles  used  to   generate  the
// templates. 18 indicates that  [0 20 ... 340] degree-rotated samples
// will be stored in the 2-D array for each scale.
static const int NUMBER_OF_ROTATION_STEPS = 18;


/* GLOBAL VARIABLES (hopefully my students won't read that) ***********************************/

// Brief object which contains methods describing keypoints with BRIEF
// descriptors.
CVLAB::BRIEF brief;

// 2-D   array   storing  detected   keypoints   for  every   template
// taken. Templates  are generated  rotating and scaling  the original
// image   for   each  rotation   angle   and   scale  determined   by
// NUMBER_OF_SCALE STEPS and NUMBER_OF_ROTATION_STEPS.
vector<cv::KeyPoint> templateKpts[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// 2-D array  storing BRIEF descriptors  of corresponding templateKpts
// elements
vector< bitset<CVLAB::DESC_LEN> > templateDescs[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// 2-D array storing  the coordinates of the corners  of the generated
// templates in original image coordinate system.
CvMat templateObjectCorners[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// Data part of the templateObjectCorners
double* templateObjectCornersData[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// The coordinates  of the keypoints  matching each other in  terms of
// Hamming  Distance between  BRIEF descriptors  of  them.  match1Data
// represents the  keypoint coordinates  of the matching  template and
// match2Data  represents  the  matching  keypoints  detected  in  the
// current frame. Elements with even indices contain x coordinates and
// those  with odd  indices  contain y  coordinates  of the  keypoints
// detected:
double match1Data[2 * MAXIMUM_NUMBER_OF_MATCHES];
double match2Data[2 * MAXIMUM_NUMBER_OF_MATCHES];


// Holds the mode of the application.

enum APPLICATION_MODE
  {
    DO_NOTHING, // Application only captures a frame and shows it
    DETECTION, // Application detects the planar object whose features are stored
    TAKING_NEW_TEMPLATE, // Application captures a new template
    END // Application quits
  } appMode;


// Holds if any template has taken before or not
bool hasAnyTemplateChosenBefore = false;


// Indicates  either all the  BRIEF descriptors  stored or  only BRIEF
// descriptors  of the  original template  will be  used  for template
// matching.
bool doDBSearch = true;


// Template image captured in RGB.
IplImage* templateImageRGBFull;


// The part of the  templateImageRGBFull which is inside the rectangle
// drawn by the user. Both in RGB and Grayscale
IplImage* templateImageRGB = NULL;
IplImage* templateImageGray = NULL;


// Last frame captured by the camera in RGB and Grayscale
IplImage* newFrameRGB;
IplImage* newFrameGray;


// Copy of the newFrameRGB for further processing.
IplImage* outputImage;


// Last frame  taken and the original  template image are  put side by
// side in order to show the result.
IplImage* sideBySideImage;


// Object used for capturing image frames
CvCapture* capture;


// Threshold value given to FAST detector:
int fastThreshold;


// Number  of the  points selected  by the  user on  the  new template
// image:
int noOfPointsPickedOnTemplateImage;


// Image coordinates  of the  points selected by  the user in  the new
// template image:
CvPoint templateCorners[2];


// Coordinate of  the top-left  corner of the  rectangle drawn  by the
// user on the new template image:
int templateROIX;
int templateROIY;


// Coordinates of the keypoints of the template which fits best to the
// frame captured by the camera.   These are the coordinates which are
// transformed back to the original template image coordinates:
double pointsOnOriginalImage[MAXIMUM_NUMBER_OF_MATCHES];


// Font which is used to write on the image
CvFont font;


// Number of the frames processed per second in the application
int fps;


// Time elapsed:
double keypointExtractTime; // by FAST detector
double bitDescTime; // to describe all keypoints with BRIEF descriptor
double matchTime; // to find the matches between 2 BRIEF descriptor vector
double hmgEstTime; // to estimate the Homography matrix between 2 images
double totalMatchTime; // to match the BRIEF descriptors of the incoming frame with all
// the BRIEF descriptors stored


// (# of Matching Keypoints / #  of the Keypoints) * 100, for the best fit:
int matchPercentage;


/****************************************** INLINE FUNCTIONS **************************************/

// Returns radian equivalent of an angle in degrees
inline float degreeToRadian(const float d)
{
  return (d / 180.0) * M_PI;
}

// Converts processor tick counts to milliseconds
inline double toElapsedTime(const double t)
{
  return t / (1e3 * cvGetTickFrequency());
}

/**************************************************************************************************/

// Function for handling keyboard inputs
void waitKeyAndHandleKeyboardInput(int timeout)
{
  // Wait for the keyboard input
  const char key = cvWaitKey(timeout);
  // Change the application mode according to the keyboard input
  switch (key) {
  case 'q': case 'Q':
    appMode = END;
    break;
  case 't': case 'T':
    if (appMode == TAKING_NEW_TEMPLATE) {
      noOfPointsPickedOnTemplateImage = 0;
      // if a template has been taken before, go back to detection of last template
      // otherwise
      appMode = hasAnyTemplateChosenBefore ? DETECTION : DO_NOTHING;
    }
    else
      appMode = TAKING_NEW_TEMPLATE;
    break;
  case 'd': case 'D':
    doDBSearch = !doDBSearch;
    break;
  }
}

// Function for handling mouse inputs
void mouseHandler(int event, int x, int y, int flags, void* params)
{
  if (appMode == TAKING_NEW_TEMPLATE) {
    templateCorners[1] = cvPoint(x, y);
    switch (event) {
    case CV_EVENT_LBUTTONDOWN:
      templateCorners[noOfPointsPickedOnTemplateImage++] = cvPoint(x, y);
      break;
    case CV_EVENT_RBUTTONDOWN:
      break;
    case CV_EVENT_MOUSEMOVE:
      if (noOfPointsPickedOnTemplateImage == 1)
	templateCorners[1] = cvPoint(x, y);
      break;
    }
  }
}

// Draws a quadrangle on an image given (u, v) coordinates, color and thickness
void drawQuadrangle(IplImage* frame,
		    const int u0, const int v0,
		    const int u1, const int v1,
		    const int u2, const int v2,
		    const int u3, const int v3,
		    const CvScalar color, const int thickness)
{
  cvLine(frame, cvPoint(u0, v0), cvPoint(u1, v1), color, thickness);
  cvLine(frame, cvPoint(u1, v1), cvPoint(u2, v2), color, thickness);
  cvLine(frame, cvPoint(u2, v2), cvPoint(u3, v3), color, thickness);
  cvLine(frame, cvPoint(u3, v3), cvPoint(u0, v0), color, thickness);
}

// Draws a quadrangle with the corners of the object detected on img
void markDetectedObject(IplImage* frame, const double * detectedCorners)
{
  drawQuadrangle(frame,
		 detectedCorners[0], detectedCorners[1],
		 detectedCorners[2], detectedCorners[3],
		 detectedCorners[4], detectedCorners[5],
		 detectedCorners[6], detectedCorners[7],
		 cvScalar(255, 255, 255), 3);
}

// Draws a plus sign on img given (x, y) coordinate
void drawAPlus(IplImage* img, const int x, const int y)
{
  cvLine(img, cvPoint(x - 5, y), cvPoint(x + 5, y), CV_RGB(255, 0, 0));
  cvLine(img, cvPoint(x, y - 5), cvPoint(x, y + 5), CV_RGB(255, 0, 0));
}

// Marks the keypoints with plus signs on img
void showKeypoints(IplImage* img, const vector<cv::KeyPoint>& kpts)
{
  for (unsigned int i = 0; i < kpts.size(); ++i)
    drawAPlus(img, kpts[i].pt.x, kpts[i].pt.y);
}

// Captures a new frame. Returns if capture is taken without problem or not.
bool takeNewFrame(void)
{
  if ((newFrameRGB = cvQueryFrame(capture)))
    cvCvtColor(newFrameRGB, newFrameGray, CV_BGR2GRAY);
  else
    return false;
  return true;
}

// Puts img1 and img2 side by side and stores into result
void putImagesSideBySide(IplImage* result, const IplImage* img1, const IplImage* img2)
{
  // widthStep of the resulting image
  const int bigWS = result->widthStep;
  // half of the widthStep of the resulting image
  const int bigHalfWS = result->widthStep >> 1;
  // widthStep of the image which will be put in the left
  const int lWS = img1->widthStep;
  // widthStep of the image which will be put in the right
  const int rWS = img2->widthStep;

  // pointer to the beginning of the left image
  char *p_big = result->imageData;
  // pointer to the beginning of the right image
  char *p_bigMiddle = result->imageData + bigHalfWS;
  // pointer to the image data which will be put in the left
  const char *p_l = img1->imageData;
  // pointer to the image data which will be put in the right
  const char *p_r = img2->imageData;

  for (int i = 0; i < FRAME_HEIGHT; ++i, p_big += bigWS, p_bigMiddle += bigWS) {
    // copy a row of the left image till the half of the resulting image
    memcpy(p_big, p_l + i*lWS, lWS);
    // copy a row of the right image from the half of the resulting image to the end of it
    memcpy(p_bigMiddle, p_r + i*rWS, rWS);
  }
}

// Marks the matching keypoints on two images which were put side by side
void showMatches(const int matchCount)
{
  const int iterationEnd = 2 * matchCount;

  for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {
    // Draw a line between matching keypoints
    cvLine(sideBySideImage,
	   cvPoint(match2Data[xCoor], match2Data[yCoor]),
	   cvPoint(pointsOnOriginalImage[xCoor] + templateROIX + FRAME_WIDTH,
		   pointsOnOriginalImage[yCoor] + templateROIY),
	   cvScalar(0, 255, 0), 1);
  }
}

// Returns whether H is a nice homography matrix or not
bool niceHomography(const CvMat * H)
{
  const double det = cvmGet(H, 0, 0) * cvmGet(H, 1, 1) - cvmGet(H, 1, 0) * cvmGet(H, 0, 1);
  if (det < 0)
    return false;

  const double N1 = sqrt(cvmGet(H, 0, 0) * cvmGet(H, 0, 0) + cvmGet(H, 1, 0) * cvmGet(H, 1, 0));
  if (N1 > 4 || N1 < 0.1)
    return false;

  const double N2 = sqrt(cvmGet(H, 0, 1) * cvmGet(H, 0, 1) + cvmGet(H, 1, 1) * cvmGet(H, 1, 1));
  if (N2 > 4 || N2 < 0.1)
    return false;

  const double N3 = sqrt(cvmGet(H, 2, 0) * cvmGet(H, 2, 0) + cvmGet(H, 2, 1) * cvmGet(H, 2, 1));
  if (N3 > 0.002)
    return false;

  return true;
}

// Rotates src around center with given angle and assigns the result to dst
void rotateImage(IplImage* dst, IplImage* src, const CvPoint2D32f& center, float angle)
{
  static CvMat *rotMat = cvCreateMat(2, 3, CV_32FC1);
  cv2DRotationMatrix(center, angle, 1.0, rotMat);
  cvWarpAffine(src, dst, rotMat);
}

// Transforms the coordinates of the keypoints of a template image whose matrix index is
// (scaleInd, rotInd) into the original template image's (scale = 1, rotation angle = 0) coordinates
void transformPointsIntoOriginalImageCoordinates(const int matchNo, const int scaleInd, const int rotInd)
{
  // Difference between the angles of two consecutive samples
  static const float ROT_ANGLE_INCREMENT = 360.0 / NUMBER_OF_ROTATION_STEPS;

  // Take the scale samples in a logarithmic base
  static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));
  const float scale = pow(k, scaleInd);

  // Center of the original image
  const float orgCenterX = templateImageGray->width / 2.0;
  const float orgCenterY = templateImageGray->height / 2.0;

  // Center of the scaled image
  const float centerX = orgCenterX * scale;
  const float centerY = orgCenterY * scale;

  // Rotation angle for the template
  const float angle = ROT_ANGLE_INCREMENT * rotInd;
  // Avoid repeatition of the trigonometric calculations
  const float cosAngle = cos(degreeToRadian(-angle));
  const float sinAngle = sin(degreeToRadian(-angle));

  const float iterationEnd = 2 * matchNo;
  for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {
    // Translate the point so that the origin is in the middle of the image
    const float translatedX = match1Data[xCoor] - centerX;
    const float translatedY = match1Data[yCoor] - centerY;

    // Rotate the point so that the angle between this template and the original template will be zero
    const float rotatedBackX = translatedX * cosAngle - translatedY * sinAngle;
    const float rotatedBackY = translatedX * sinAngle + translatedY * cosAngle;

    // Scale the point so that the size of this template will be equal to the original one
    pointsOnOriginalImage[xCoor] = rotatedBackX / scale + orgCenterX;
    pointsOnOriginalImage[yCoor] = rotatedBackY / scale + orgCenterY;
  }
}

// Estimates the fps of the application
void fpsCalculation(void)
{
  static int64 currentTime, lastTime = cvGetTickCount();
  static int fpsCounter = 0;
  currentTime = cvGetTickCount();
  ++fpsCounter;
  
  // If 1 second has passed since the last FPS estimation, update the fps
  if (currentTime - lastTime > 1e6 * cvGetTickFrequency()) {
    fps = fpsCounter;
    lastTime = currentTime;
    fpsCounter = 0;
  }
}

// Writes the statistics showing the performance of the application to img
void showOutput(IplImage* img)
{
  static char text[256];

  if (appMode != TAKING_NEW_TEMPLATE) {
    sprintf(text, "FPS: %d", fps);
    cvPutText(img, text, cvPoint(10, 30), &font, cvScalar(255, 0, 0));

    sprintf(text, "KP Extract: %f", toElapsedTime(keypointExtractTime));
    cvPutText(img, text, cvPoint(10, 50), &font, cvScalar(255, 0, 0));

    sprintf(text, "Bit Desc: %f", toElapsedTime(bitDescTime));
    cvPutText(img, text, cvPoint(10, 70), &font, cvScalar(255, 0, 0));

    sprintf(text, "Match Time: %f", toElapsedTime(matchTime));
    cvPutText(img, text, cvPoint(10, 90), &font, cvScalar(255, 0, 0));

    sprintf(text, "Total Matching Time: %f", toElapsedTime(totalMatchTime));
    cvPutText(img, text, cvPoint(10, 110), &font, cvScalar(255, 0, 0));

    sprintf(text, "RANSAC: %f", toElapsedTime(hmgEstTime));
    cvPutText(img, text, cvPoint(10, 130), &font, cvScalar(255, 0, 0));

    sprintf(text, "Match Percentage: %d%%", matchPercentage);
    cvPutText(img, text, cvPoint(10, 150), &font, cvScalar(255, 0, 0));
  }
  
  cvShowImage("BRIEF", img);
}

// Detect keypoints of img with FAST and store them to kpts given the threshold kptDetectorThreshold.
int extractKeypoints(vector< cv::KeyPoint >& kpts, int kptDetectorThreshold, IplImage* img)
{
  CvRect r = cvRect(CVLAB::IMAGE_PADDING_LEFT, CVLAB::IMAGE_PADDING_TOP,
		    CVLAB::SUBIMAGE_WIDTH(img->width), CVLAB::SUBIMAGE_HEIGHT(img->height));

  // Don't detect keypoints on the image borders:
  cvSetImageROI(img, r);

  // Use FAST corner detector to detect the image keypoints
  cv::FAST(img, kpts, kptDetectorThreshold, true);

  // Get the borders back:
  cvResetImageROI(img);

  // Transform the points to their actual image coordinates:
  for (unsigned int i = 0, sz = kpts.size(); i < sz; ++i)
    kpts[i].pt.x += CVLAB::IMAGE_PADDING_LEFT, kpts[i].pt.y += CVLAB::IMAGE_PADDING_TOP;

  return kpts.size();
}

// Tries to find a threshold for FAST that gives a number of keypoints between lowerBound and upperBound:
int chooseFASTThreshold(const IplImage* img, const int lowerBound, const int upperBound)
{
  static vector<cv::KeyPoint> kpts;

  int left = 0;
  int right = 255;
  int currentThreshold = 128;
  int currentScore = 256;

  IplImage* copyImg = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
  cvCopyImage(img, copyImg);

  while (currentScore < lowerBound || currentScore > upperBound) {
    currentScore = extractKeypoints(kpts, currentThreshold, copyImg);

    if (lowerBound > currentScore) {
      // we look for a lower threshold to increase the number of corners:
      right = currentThreshold;
      currentThreshold = (currentThreshold + left) >> 1;
      if (right == currentThreshold)
	break;
    } else {
      // we look for a higher threshold to decrease the number of corners:
      left = currentThreshold;
      currentThreshold = (currentThreshold + right) >> 1;
      if (left == currentThreshold)
	break;
    }
  }
  cvReleaseImage(&copyImg);

  return currentThreshold;
}

// Saves the coordinates of the corners of the rectangle drawn by the user when
// capturing a new template image
void saveCornerCoors(void)
{
  const double templateWidth = templateImageGray->width;
  const double templateHeight = templateImageGray->height;

  double* corners = templateObjectCornersData[0][0];
  corners[0] = 0;
  corners[1] = 0;
  corners[2] = templateWidth;
  corners[3] = 0;
  corners[4] = templateWidth;
  corners[5] = templateHeight;
  corners[6] = 0;
  corners[7] = templateHeight;
}

// Saves the image inside the rectangle drawn by the user as the new template image.
// Returns if it can be used as a new template or not.
bool saveNewTemplate(void)
{
  // Calculate the size of the new template
  const int templateWidth = templateCorners[1].x - templateCorners[0].x;
  const int templateHeight = templateCorners[1].y - templateCorners[0].y;

  // If the size of of the new template is illegal, return false
  if ((SMALLEST_SCALE_CHANGE * templateWidth) < CVLAB::IMAGE_PADDING_TOTAL ||
      (SMALLEST_SCALE_CHANGE * templateHeight) < CVLAB::IMAGE_PADDING_TOTAL)
    return false;

  // Store the upper left corner coordinate of the rectangle (ROI)
  templateROIX = templateCorners[0].x, templateROIY = templateCorners[0].y;

  const CvSize templateSize = cvSize(templateWidth, templateHeight);
  const CvRect templateRect = cvRect(templateCorners[0].x, templateCorners[0].y, 
				     templateWidth, templateHeight);
  
  // Store the original version of the new template(all image)
  cvCopyImage(newFrameRGB, templateImageRGBFull);

  cvReleaseImage(&templateImageRGB);
  templateImageRGB = cvCreateImage(templateSize, IPL_DEPTH_8U, 3);

  cvReleaseImage(&templateImageGray);
  templateImageGray = cvCreateImage(templateSize, IPL_DEPTH_8U, 1);

  // Store a Grayscale version of the new template(only ROI)
  cvSetImageROI(newFrameGray, templateRect);
  cvCopyImage(newFrameGray, templateImageGray);
  cvResetImageROI(newFrameGray);


  // Store an RGB version of the new template(only ROI)
  cvSetImageROI(newFrameRGB, templateRect);
  cvCopyImage(newFrameRGB, templateImageRGB);
  cvResetImageROI(newFrameRGB);

  saveCornerCoors();

  return true;
}

// Finds the coordinates of the corners of the template image after scaling and rotation applied
void estimateCornerCoordinatesOfNewTemplate(int scaleInd, int rotInd, float scale, float angle)
{
  static double* corners = templateObjectCornersData[0][0];

  // Center of the original image
  const float orgCenterX = templateImageGray->width / 2.0, orgCenterY = templateImageGray->height / 2.0;
  // Center of the scaled image
  const float centerX = orgCenterX * scale, centerY = orgCenterY * scale;

  const float cosAngle = cos(degreeToRadian(angle));
  const float sinAngle = sin(degreeToRadian(angle));

  for (int xCoor = 0, yCoor = 1; xCoor < 8; xCoor += 2, yCoor += 2) {
    // Scale the point and translate it so that the origin is in the middle of the image
    const float resizedAndTranslatedX = (corners[xCoor] * scale) - centerX,
      resizedAndTranslatedY = (corners[yCoor] * scale) - centerY;

    // Rotate the point with the given angle
    templateObjectCornersData[scaleInd][rotInd][xCoor] =
      (resizedAndTranslatedX * cosAngle - resizedAndTranslatedY * sinAngle) + centerX;
    templateObjectCornersData[scaleInd][rotInd][yCoor] =
      (resizedAndTranslatedX * sinAngle + resizedAndTranslatedY * cosAngle) + centerY;
  }
}

// Generates new templates with different scales and orientations and stores their keypoints and
// BRIEF descriptors.
void learnTemplate(void)
{
  static const float ROT_ANGLE_INCREMENT = 360.0 / NUMBER_OF_ROTATION_STEPS;
  static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));

  // Estimate a feasible threshold value for FAST keypoint detector
  fastThreshold = chooseFASTThreshold(templateImageGray, 200, 250);

  // For every scale generate templates
  for (int scaleInd = 0; scaleInd < NUMBER_OF_SCALE_STEPS; ++scaleInd) {
    // Calculate the template size in a log basis
    const float currentScale = pow(k, scaleInd);

    // Scale the image
    IplImage* scaledTemplateImg = cvCreateImage(cvSize(templateImageGray->width * currentScale,
						       templateImageGray->height * currentScale),
						IPL_DEPTH_8U, 1);
    cvResize(templateImageGray, scaledTemplateImg);

    const CvPoint2D32f center = cvPoint2D32f(scaledTemplateImg->width >> 1, scaledTemplateImg->height >> 1);

    // For a given scale, generate templates with several rotations
    float currentAngle = 0.0;
    for (int rotInd = 0; rotInd < NUMBER_OF_ROTATION_STEPS; ++rotInd, currentAngle += ROT_ANGLE_INCREMENT) {
      // Rotate the image
      IplImage* rotatedImage = cvCreateImage(cvGetSize(scaledTemplateImg),
					     scaledTemplateImg->depth,
					     scaledTemplateImg->nChannels);
      rotateImage(rotatedImage, scaledTemplateImg, center, -currentAngle);

      // Detect FAST keypoints
      extractKeypoints(templateKpts[scaleInd][rotInd], fastThreshold, rotatedImage);

      // Describe the keypoints with BRIEF descriptors
      brief.getBriefDescriptors(templateDescs[scaleInd][rotInd],
				templateKpts[scaleInd][rotInd],
				rotatedImage);

      // Store the scaled and rotated template corner coordinates
      estimateCornerCoordinatesOfNewTemplate(scaleInd, rotInd, currentScale, currentAngle);

      cvReleaseImage(&rotatedImage);
    }
    cvReleaseImage(&scaledTemplateImg);
  }
}

// Manages the capture of the new template image according to the points picked by the user
void takeNewTemplateImage(void)
{
  cvCopyImage(newFrameRGB, outputImage);
  switch (noOfPointsPickedOnTemplateImage) {
  case 1:
    cvRectangle(outputImage, templateCorners[0], templateCorners[1], cvScalar(0, 255, 0), 3);
    break;
  case 2:
    if (saveNewTemplate()) {
      learnTemplate();
      appMode = DETECTION;
      hasAnyTemplateChosenBefore = true;
    }
    noOfPointsPickedOnTemplateImage = 0;
    break;
  default:
    break;
  }
}

// Matches Brief descriptors descs1 and descs2 in terms of Hamming Distance.
int matchDescriptors(
		     CvMat& match1, CvMat& match2,
		     const vector< bitset<CVLAB::DESC_LEN> > descs1,
		     const vector< bitset<CVLAB::DESC_LEN> > descs2,
		     const vector<cv::KeyPoint>& kpts1,
		     const vector<cv::KeyPoint>& kpts2)
{
  // Threshold value for matches.
  static const int MAX_MATCH_DISTANCE = 50;

  int numberOfMatches = 0;
  // Index of the best BRIEF descriptor match on descs2
  int bestMatchInd2 = 0;

  // For every BRIEF descriptor in descs1 find the best fitting BRIEF descriptor in descs2
  for (unsigned int i = 0; i < descs1.size() && numberOfMatches < MAXIMUM_NUMBER_OF_MATCHES; ++i) {
    int minDist = CVLAB::DESC_LEN;
    
    for (unsigned int j = 0; j < descs2.size(); ++j) {
      const int dist = CVLAB::HAMMING_DISTANCE(descs1[i], descs2[j]);
      
      // If dist is less than the optimum one observed so far, the new optimum one is current BRIEF descriptor
      if (dist < minDist) {
	minDist = dist;
	bestMatchInd2 = j;
      }
    }
    // If the Hamming Distance is greater than the threshold, ignore this match
    if (minDist > MAX_MATCH_DISTANCE)
      continue;

    // Save the matching keypoint coordinates
    const int xInd = 2 * numberOfMatches;
    const int yInd = xInd + 1;

    match1Data[xInd] = kpts1[i].pt.x;
    match1Data[yInd] = kpts1[i].pt.y;

    match2Data[xInd] = kpts2[bestMatchInd2].pt.x;
    match2Data[yInd] = kpts2[bestMatchInd2].pt.y;
    
    numberOfMatches++;
  }

  if (numberOfMatches > 0) {
    cvInitMatHeader(&match1, numberOfMatches, 2, CV_64FC1, match1Data);
    cvInitMatHeader(&match2, numberOfMatches, 2, CV_64FC1, match2Data);
  }

  return numberOfMatches;
}

// Initializes the application
void init(void)
{
  // Seed the random number generator
  srand(time(NULL));

  // In the beginning, only capture a frame and show it to the user
  appMode = DO_NOTHING;

  capture = cvCaptureFromCAM(0); // capture from video device #0
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

  // Memory Allocations
  newFrameGray = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 1);
  outputImage = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
  templateImageRGBFull = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
  sideBySideImage = cvCreateImage(cvSize(2 * FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
  templateImageRGB = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 3);
  templateImageGray = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 1);
  for (int s = 0; s < NUMBER_OF_SCALE_STEPS; s++) {
    for (int r = 0; r < NUMBER_OF_ROTATION_STEPS; r++) {
      templateObjectCornersData[s][r] = new double[8];
      templateObjectCorners[s][r] = cvMat(1, 4, CV_64FC2, templateObjectCornersData[s][r]);
    }
  }

  cvNamedWindow("BRIEF", CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback("BRIEF", mouseHandler, NULL);

  cvInitFont(&font, CV_FONT_HERSHEY_PLAIN | CV_FONT_ITALIC, 1, 1, 0, 1);
}

// Detects the template object in the incoming frame
void doDetection(void)
{
  // Variables for elapsed time estimations
  static int64 startTime, endTime;

  // Homography Matrix
  static CvMat* H = cvCreateMat(3, 3, CV_64FC1);

  // Corners of the detected planar object
  static double detectedObjCornersData[8];
  static CvMat detectedObjCorners = cvMat(1, 4, CV_64FC2, detectedObjCornersData);

  // Keypoints of the incoming frame
  vector<cv::KeyPoint> kpts;
  
  // BRIEF descriptors of kpts
  vector< bitset<CVLAB::DESC_LEN> > descs;

  // Coordinates of the matching keypoints
  CvMat match1, match2;

  float maxRatio = 0.0;
  int maxScaleInd = 0;
  int maxRotInd = 0;
  int maximumNumberOfMatches = 0;

  // If !doDBSearch then only try to match the original template image
  const int dbScaleSz = doDBSearch ? NUMBER_OF_SCALE_STEPS : 1;
  const int dbRotationSz = doDBSearch ? NUMBER_OF_ROTATION_STEPS : 1;

  startTime = cvGetTickCount();
  // Detect the FAST keypoints of the incoming frame
  extractKeypoints(kpts, fastThreshold, newFrameGray);
  endTime = cvGetTickCount();
  keypointExtractTime = endTime - startTime;


  startTime = cvGetTickCount();
  // Describe the keypoints with BRIEF descriptors
  brief.getBriefDescriptors(descs, kpts, newFrameGray);
  endTime = cvGetTickCount();
  bitDescTime = endTime - startTime;

  startTime = cvGetTickCount();
  // Search through all the templates
  for (int scaleInd = 0; scaleInd < dbScaleSz; ++scaleInd) {
    for (int rotInd = 0; rotInd < dbRotationSz; ++rotInd) {
      const int numberOfMatches = matchDescriptors(match1, match2,
						   templateDescs[scaleInd][rotInd], descs,
						   templateKpts[scaleInd][rotInd], kpts);

      // Since RANSAC needs at least 4 points, ignore this match
      if (numberOfMatches < 4)
	continue;

      // Save the matrix index of the best fitting template to the incoming frame
      const float currentRatio = float(numberOfMatches) / templateKpts[scaleInd][rotInd].size();
      if (currentRatio > maxRatio) {
	maxRatio = currentRatio;
	maxScaleInd = scaleInd;
	maxRotInd = rotInd;
	maximumNumberOfMatches = numberOfMatches;
      }
    }
  }
  endTime = cvGetTickCount();
  totalMatchTime = endTime - startTime;


  matchPercentage = int(maxRatio * 100.0);

  if (maximumNumberOfMatches > 3) {
    startTime = cvGetTickCount();
    // Match the best fitting template's BRIEF descriptors with the incoming frame
    matchDescriptors(match1, match2,
		     templateDescs[maxScaleInd][maxRotInd], descs,
		     templateKpts[maxScaleInd][maxRotInd], kpts);
    endTime = cvGetTickCount();
    matchTime = endTime - startTime;

    // Calculate the homography matrix via RANSAC
    cvFindHomography(&match1, &match2, H, CV_RANSAC, 10, 0);

    // If H is not a feasible homography matrix, ignore it
    if (niceHomography(H)) {
      startTime = cvGetTickCount();
      // Transform the coordinates of the corners of the template into image coordinates
      cvPerspectiveTransform(&templateObjectCorners[maxScaleInd][maxRotInd], &detectedObjCorners, H);
      endTime = cvGetTickCount();
      hmgEstTime = endTime - startTime;

      // Draw the detected object on the image
      markDetectedObject(sideBySideImage, detectedObjCornersData);

      // Scale and rotate the coordinates of the template keypoints to transform them into the original
      // template image's coordinates to show the matches
      transformPointsIntoOriginalImageCoordinates(maximumNumberOfMatches, maxScaleInd, maxRotInd);
    }
  }

  // Mark the keypoints detected with plus signs:
  showKeypoints(sideBySideImage, kpts);

  // Indicate the matches with lines:
  showMatches(maximumNumberOfMatches);
}

// Main loop of the program
void run(void)
{
  while (true) {
    IplImage* result = outputImage;

    fpsCalculation();

    switch (appMode) {
    case TAKING_NEW_TEMPLATE:
      takeNewTemplateImage();
      break;
    case DETECTION:
      takeNewFrame();
      cvCopyImage(newFrameRGB, outputImage);
      putImagesSideBySide(sideBySideImage, newFrameRGB, templateImageRGBFull);
      doDetection();
      result = sideBySideImage;
      break;
    case DO_NOTHING:
      takeNewFrame();
      cvCopyImage(newFrameRGB, outputImage);
      break;
    case END:
      return;
    default:
      break;
    }
    showOutput(result);
    waitKeyAndHandleKeyboardInput(10);
  }
}

int main(void)
{
  init();

  cout << "Press:" << endl;
  cout << " 't' to capture a new template;" << endl;
  cout << " 'd' to enable/disable scale and rotation invariance;" << endl;
  cout << " 'q' to quit." << endl;
  run();

  return 0;
}

