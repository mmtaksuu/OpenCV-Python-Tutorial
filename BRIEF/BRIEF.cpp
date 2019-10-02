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
#include <ctime>
#include <iostream>

using namespace std;
using namespace CVLAB;

BRIEF::BRIEF(void)
{
  // Allocates memory for 'testLocations' and 'intensityPairs':
  testLocations = vector< pair<cv::Point2i, cv::Point2i> >(DESC_LEN);
  integralImage = cvCreateMat(1, 1, CV_32SC1);

  // Picks the pixel pairs for intensity comparison tests:
  pickTestLocations();
}

BRIEF::~BRIEF(void)
{
  // Free allocated memory segments:
  testLocations.clear();
  cvReleaseMat(&integralImage);
}

void BRIEF::getBriefDescriptor(bitset<DESC_LEN>& desc, cv::KeyPoint kpt, IplImage* img)
{
  // Calculate the width step of the integral image
  int inWS = integralImage->step / CV_ELEM_SIZE(integralImage->type);

  // Hold the pointer of the data part of integral image matrix with 'iD'
  int* iD = integralImage->data.i;

  // Iterate over test location pairs
  for (int i = 0; i < DESC_LEN; ++i) {
    // tL holds the current test location pairs
    const pair<cv::Point2i, cv::Point2i>& tL = testLocations[i];

    // Transform the testpoints from patch coordinates to the image coordinate
    const cv::Point2i p1 = CV_POINT_PLUS(kpt.pt, tL.first);
    const cv::Point2i p2 = CV_POINT_PLUS(kpt.pt, tL.second);

    // Use the integral image to compare the average intensities around the 2 test locations:
    const int intensity1 =
      GET_PIXEL_NW(iD, p1, inWS) - GET_PIXEL_NE(iD, p1, inWS) -
      GET_PIXEL_SW(iD, p1, inWS) + GET_PIXEL_SE(iD, p1, inWS);

    const int intensity2 =
      GET_PIXEL_NW(iD, p2, inWS) - GET_PIXEL_NE(iD, p2, inWS) -
      GET_PIXEL_SW(iD, p2, inWS) + GET_PIXEL_SE(iD, p2, inWS);
    
    desc[i] = intensity1 < intensity2;
  }
}

void BRIEF::getBriefDescriptors(vector< bitset<DESC_LEN> >& descriptors, const vector<cv::KeyPoint>& kpts,
				IplImage* img)
{
  // Make sure that input image contains only one color channel:
  assert(img->nChannels == 1);

  // Check whether all the keypoints are inside the subimage:
  assert(validateKeypoints(kpts, img->width, img->height));

  // If memory allocated in 'descriptors' is not enough, then resize it:
  descriptors.resize(kpts.size());

  // Allocate space for the integral image:
  allocateIntegralImage(img);

  // Calculate the integral image:
  cvIntegral(img, integralImage);

  // Iterate over keypoints:
  for (unsigned int i = 0; i < kpts.size(); ++i)
    getBriefDescriptor(descriptors[i], kpts[i], img);
}

void BRIEF::pickTestLocations(void)
{
  // Pick test locations totally randomly in a way that the locations are inside the patch. Note that
  // number of the tests is equal to the length of the descriptor.
  for (int i = 0; i < DESC_LEN; ++i) {
    pair<cv::Point2i, cv::Point2i>& tL = testLocations[i];

    tL.first.x  = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
    tL.first.y  = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
    tL.second.x = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
    tL.second.y = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
  }
}

bool BRIEF::isKeypointInsideSubImage(const cv::KeyPoint& kpt, const int width, const int height)
{
  return
    SUBIMAGE_LEFT <= kpt.pt.x  &&  kpt.pt.x < SUBIMAGE_RIGHT(width) &&
    SUBIMAGE_TOP  <= kpt.pt.y  &&  kpt.pt.y < SUBIMAGE_BOTTOM(height);
}

bool BRIEF::validateKeypoints(const vector<cv::KeyPoint>& kpts, int im_w, int im_h)
{
  for (unsigned int i = 0; i < kpts.size(); ++i)
    if ( !isKeypointInsideSubImage(kpts[i], im_w, im_h) )
      return false;

  return true;
}

void BRIEF::allocateIntegralImage(const IplImage* img)
{
  const int im_w_1 = img->width + 1, im_h_1 = img->height + 1;

  if (im_w_1 != integralImage->width && im_h_1 != integralImage->height) {
    cvReleaseMat(&integralImage);
    integralImage = cvCreateMat(im_h_1, im_w_1, CV_32SC1);
  }
}
