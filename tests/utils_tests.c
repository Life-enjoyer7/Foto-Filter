#include "utils_tests.h"
#include <stdio.h>
#include <stdlib.h>

int g_test_failures = 0;

const char *strategy_names[7] = {
    "Sequential",   "Pixelwise",    "By Rows",       "By Cols",
    "Blocks 32x32", "Blocks 64x64", "Blocks 128x128"};

IplImage *createRandomImage(int w, int h) {
  IplImage *img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
  int step = img->widthStep;
  int channels = img->nChannels;
  unsigned char *data = (unsigned char *)img->imageData;

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      unsigned char *p = data + y * step + x * channels;
      p[0] = (unsigned char)(rand() % 256);
      p[1] = (unsigned char)(rand() % 256);
      p[2] = (unsigned char)(rand() % 256);
    }
  return img;
}

int randomImageDim(void) {
  int r = rand() % 10;
  if (r == 0)
    return 1;
  if (r == 1)
    return 2;
  if (r == 2)
    return 3;
  return 4 + rand() % 57;
}

int randomFilterIdForDim(int minDim) {
  const int f3x3[] = {0, 2, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 26};
  const int f5x5[] = {1, 3, 5, 6, 7, 10, 13, 21, 22, 24, 25};
  const int f7x7[] = {23};
  const int f9x9[] = {4};

  int pool[27];
  int n = 0;
  for (int i = 0; i < 14; i++)
    pool[n++] = f3x3[i];
  if (minDim >= 2)
    for (int i = 0; i < 11; i++)
      pool[n++] = f5x5[i];
  if (minDim >= 3)
    pool[n++] = f7x7[0];
  if (minDim >= 4)
    pool[n++] = f9x9[0];

  return pool[rand() % n];
}

IplImage *referenceApplyFilter(const IplImage *src, const Filter *f) {
  int padW = f->width / 2;
  int padH = f->height / 2;

  CvMat *kernel = cvCreateMat(f->height, f->width, CV_64FC1);
  for (int y = 0; y < f->height; y++)
    for (int x = 0; x < f->width; x++)
      cvmSet(kernel, y, x, f->matrix[y][x] * f->factor);

  IplImage *padded =
      cvCreateImage(cvSize(src->width + 2 * padW, src->height + 2 * padH),
                    IPL_DEPTH_8U, src->nChannels);
  cvCopyMakeBorder(src, padded, cvPoint(padW, padH), IPL_BORDER_WRAP,
                   cvScalarAll(0));

  IplImage *filtered64 =
      cvCreateImage(cvGetSize(padded), IPL_DEPTH_64F, src->nChannels);
  cvFilter2D(padded, filtered64, kernel, cvPoint(-1, -1));

  IplImage *result =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, src->nChannels);
  int step = result->widthStep;
  int channels = src->nChannels;
  const double *fdata = (const double *)filtered64->imageData;
  int fstep = filtered64->widthStep / (int)sizeof(double);
  unsigned char *dst_data = (unsigned char *)result->imageData;

  for (int y = 0; y < src->height; y++)
    for (int x = 0; x < src->width; x++) {
      unsigned char *out = dst_data + y * step + x * channels;
      for (int c = 0; c < channels; c++) {
        double val =
            fdata[(y + padH) * fstep + (x + padW) * channels + c] + f->bias;
        int v = (int)val;
        v = v < 0 ? 0 : (v > 255 ? 255 : v);
        out[c] = (unsigned char)v;
      }
    }

  cvReleaseMat(&kernel);
  cvReleaseImage(&padded);
  cvReleaseImage(&filtered64);
  return result;
}

int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance) {
  if (a->width != b->width || a->height != b->height ||
      a->nChannels != b->nChannels)
    return 0;

  int step = a->widthStep;
  int channels = a->nChannels;
  const unsigned char *da = (const unsigned char *)a->imageData;
  const unsigned char *db = (const unsigned char *)b->imageData;

  for (int y = 0; y < a->height; y++)
    for (int x = 0; x < a->width; x++) {
      const unsigned char *pa = da + y * step + x * channels;
      const unsigned char *pb = db + y * step + x * channels;
      for (int c = 0; c < 3; c++)
        if (abs((int)pa[c] - (int)pb[c]) > tolerance)
          return 0;
    }
  return 1;
}
