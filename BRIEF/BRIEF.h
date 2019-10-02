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

#ifndef __BRIEF_H__
#define __BRIEF_H__

#include <vector>
#include <bitset>
#include <cv.h>
#include <highgui.h>

using namespace std;

namespace CVLAB {

  // Length of the BRIEF descriptor:
  static const int DESC_LEN = 256;

  // Size of the area in which the tests are computed:
  static const int PATCH_SIZE = 37;

  // Box kernel size used for smoothing. Must be an odd positive integer:
  static const int KERNEL_SIZE = 9;


  // Size of the area surrounding feature point
  static const int PATCH_SIZE_2 = PATCH_SIZE * PATCH_SIZE;
  // Half of the patch size
  static const int HALF_PATCH_SIZE = PATCH_SIZE >> 1;
  // Area of the smoothing kernel
  static const int KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE;
  // Half of the kernel size
  static const int HALF_KERNEL_SIZE = KERNEL_SIZE >> 1;

  /***************************************************************************************************
   *
   *  +++++++++++++++++++
   *  +-a-            | +
   *  +               b +       a = IMAGE_PADDING_LEFT
   *  +               | +       b = IMAGE_PADDING_TOP
   *  +   +++++++++++   +       The area inside is the subimage we look for the keypoints such that
   *  +   +         +   +       all the test locations we pick and the corners of the smoothing kernels
   *  +   +         +   +       we apply to these test locations are guaranteed to be inside the image.
   *  +   +         +   +       Note that in our implementation a = b
   *  +   +         +   +
   *  +   +++++++++++   +
   *  + |            -a-+
   *  + b               +
   *  + |               +
   *  +++++++++++++++++++
   *  Figure 1
   *
   **************************************************************************************************/

  // See figure above:
  static const int IMAGE_PADDING_TOP = HALF_KERNEL_SIZE + HALF_PATCH_SIZE;
  static const int IMAGE_PADDING_LEFT = IMAGE_PADDING_TOP;
  static const int IMAGE_PADDING_TOTAL = IMAGE_PADDING_TOP << 1;
  static const int IMAGE_PADDING_RIGHT = IMAGE_PADDING_LEFT;
  static const int IMAGE_PADDING_BOTTOM = IMAGE_PADDING_TOP;
  static const int SUBIMAGE_LEFT = IMAGE_PADDING_LEFT;
  static const int SUBIMAGE_TOP = IMAGE_PADDING_TOP;

  // Returns the Hamming Distance between two BRIEF descriptors
  inline int HAMMING_DISTANCE(const bitset<DESC_LEN>& d1, const bitset<DESC_LEN>& d2)
  {
    return (d1 ^ d2).count();
  }

  // Returns the width of the subimage shown in the figure above given the original image width:
  inline int SUBIMAGE_WIDTH(const int width)
  {
    return width - IMAGE_PADDING_TOTAL;
  }

  // Returns the width of the subimage shown in the figure above given the original image width:
  inline int SUBIMAGE_HEIGHT(const int height)
  {
    return height - IMAGE_PADDING_TOTAL;
  }

  // Returns the x-coordinate of the right edge of the subimage
  inline int SUBIMAGE_RIGHT(const int width) 
  {
    return width - IMAGE_PADDING_RIGHT;
  }

  // Returns the y-coordinate of the bottom edge of the subimage
  inline int SUBIMAGE_BOTTOM(const int height) 
  {
    return height - IMAGE_PADDING_BOTTOM;
  }
  
  // Returns pd[row][column]
  inline int GET_MATRIX_DATA(const int* pD, const int row, int column, const int wS)
  {
    return *(pD + (row * wS) + column);
  }

  // Returns the value of north-west corner of the kernel
  inline int GET_PIXEL_NW(const int* pD, const cv::Point2i& point, const int wS)
  {
    return GET_MATRIX_DATA(pD, point.y - HALF_KERNEL_SIZE, point.x - HALF_KERNEL_SIZE, wS);
  }

  // Returns the value of north-east corner of the kernel
  inline int GET_PIXEL_NE(const int* pD, const cv::Point2i& point, const int wS)
  {
    return GET_MATRIX_DATA(pD, point.y - HALF_KERNEL_SIZE, point.x + HALF_KERNEL_SIZE, wS);
  }

  // Returns the value of south-west corner of the kernel
  inline int GET_PIXEL_SW(const int* pD, const cv::Point2i& point, const int wS)
  {
    return GET_MATRIX_DATA(pD, point.y + HALF_KERNEL_SIZE, point.x - HALF_KERNEL_SIZE, wS);
  }

  // Returns the value of south-east corner of the kernel
  inline int GET_PIXEL_SE(const int* pD, const cv::Point2i& point, const int wS)
  {
    return GET_MATRIX_DATA(pD, point.y + HALF_KERNEL_SIZE, point.x + HALF_KERNEL_SIZE, wS);
  }

  // Returns a cv::Point2i which is the sum of two cv::Point2i
  inline cv::Point2i CV_POINT_PLUS(const cv::Point2i& p, const cv::Point2i& delta)
  {
    return cv::Point2i(p.x + delta.x, p.y + delta.y);
  }

  // A class which represents the operations of BRIEF keypoint descriptor
  class BRIEF {
  public:
    // The constructor only pre-computes the tests locations:
    BRIEF(void);

    // Destructor method:
    virtual ~BRIEF();

    // Given keypoint kpt and image img, returns the BRIEF descriptor desc
    void getBriefDescriptor(bitset<DESC_LEN>& desc, cv::KeyPoint kpt, IplImage* img);

    // Given keypoints kpts and image img, returns BRIEF descriptors descs
    void getBriefDescriptors(vector< bitset<DESC_LEN> >& descriptors, const vector<cv::KeyPoint>& kpts, IplImage* img);

  private:
    // Choose the test locations arbitrarily
    void pickTestLocations(void);

    // Allocate space for storing integral image
    void allocateIntegralImage(const IplImage* img);

    // Checks if the tests locations for the keypoints in kpts lie inside an im_w x im_h image:
    bool validateKeypoints(const vector< cv::KeyPoint >& kpts, int im_w, int im_h);
    
    // Returns true if kpt is inside the subimage
    bool isKeypointInsideSubImage(const cv::KeyPoint& kpt, const int width, const int height);

    // Test locations used for intensity comparison tests
    vector< pair<cv::Point2i, cv::Point2i> > testLocations;

    CvMat* integralImage;
  };

};

#endif
