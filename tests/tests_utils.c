#include "tests_utils.h"
#include "../src/main_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int g_test_failures = 0;

const char *const imagePaths[15] = {
    "images/bugatti_1200x600.jpg",    "images/bugatti_1536x2048.jpg",
    "images/bugatti_3275x4096.jpg",   "images/ferari_320x320.jpg",
    "images/ferrari_2560x1440.jpg",   "images/ford_1080x1080.jpg",
    "images/lambo_236x236.jpg",       "images/lambo_1080x1349.jpg",
    "images/maseratti_2048x2048.jpg", "images/mustang_736x736.jpg",
    "images/sportcar_474x503.jpg",    "images/sportcar_736x981.jpg",
    "images/sportcar_3823x4237.jpg",  "images/bugatti_big_size.jpg",
    "images/car_1200x687.jpeg"};

const char *const imageNames[15] = {
    "bugatti_1200x600.jpg",    "bugatti_1536x2048.jpg",
    "bugatti_3275x4096.jpg",   "ferari_320x320.jpg",
    "ferrari_2560x1440.jpg",   "ford_1080x1080.jpg",
    "lambo_236x236.jpg",       "lambo_1080x1349.jpg",
    "maseratti_2048x2048.jpg", "mustang_736x736.jpg",
    "sportcar_474x503.jpg",    "sportcar_736x981.jpg",
    "sportcar_3823x4237.jpg",  "bugatti_big_size.jpg",
    "car_1200x687.jpeg"};

#define BENCH_REPEAT 3

double timeFilterMs(FilterFn fn, const IplImage *src, IplImage *dst,
                    const Filter *f) {
  fn(src, dst, f);

  double total = 0.0;
  for (int i = 0; i < BENCH_REPEAT; i++) {
    double start = get_time_ms();
    fn(src, dst, f);
    double end = get_time_ms();
    total += (end - start);
  }
  return total / BENCH_REPEAT;
}

double timeTwoFiltersMs(FilterFn fn, const IplImage *srcA, IplImage *dstA,
                        const Filter *fA, const IplImage *srcB, IplImage *dstB,
                        const Filter *fB) {
  fn(srcA, dstA, fA);
  fn(srcB, dstB, fB);

  double total = 0.0;
  for (int i = 0; i < BENCH_REPEAT; i++) {
    double start = get_time_ms();
    fn(srcA, dstA, fA);
    fn(srcB, dstB, fB);
    double end = get_time_ms();
    total += (end - start);
  }
  return total / BENCH_REPEAT;
}

int imagesEqual(const IplImage *a, const IplImage *b) {
  if (a->width != b->width || a->height != b->height ||
      a->nChannels != b->nChannels)
    return 0;

  int step = a->widthStep;
  int channels = a->nChannels;
  const unsigned char *data_a = (const unsigned char *)a->imageData;
  const unsigned char *data_b = (const unsigned char *)b->imageData;

  for (int y = 0; y < a->height; y++) {
    for (int x = 0; x < a->width; x++) {
      const unsigned char *pa = data_a + y * step + x * channels;
      const unsigned char *pb = data_b + y * step + x * channels;
      if (pa[0] != pb[0] || pa[1] != pb[1] || pa[2] != pb[2])
        return 0;
    }
  }
  return 1;
}

int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance,
                      int *max_diff_out) {
  if (a->width != b->width || a->height != b->height ||
      a->nChannels != b->nChannels)
    return 0;

  int step = a->widthStep;
  int channels = a->nChannels;
  const unsigned char *da = (const unsigned char *)a->imageData;
  const unsigned char *db = (const unsigned char *)b->imageData;
  int max_diff = 0;
  int ok = 1;

  for (int y = 0; y < a->height; y++) {
    for (int x = 0; x < a->width; x++) {
      const unsigned char *pa = da + y * step + x * channels;
      const unsigned char *pb = db + y * step + x * channels;
      for (int c = 0; c < 3; c++) {
        int diff = abs((int)pa[c] - (int)pb[c]);
        if (diff > max_diff)
          max_diff = diff;
        if (diff > tolerance)
          ok = 0;
      }
    }
  }

  if (max_diff_out)
    *max_diff_out = max_diff;
  return ok;
}

int imageIsBlack(const IplImage *img) {
  int step = img->widthStep;
  int channels = img->nChannels;
  const unsigned char *data = (const unsigned char *)img->imageData;

  for (int y = 0; y < img->height; y++) {
    for (int x = 0; x < img->width; x++) {
      const unsigned char *pixel = data + y * step + x * channels;
      if (pixel[0] || pixel[1] || pixel[2])
        return 0;
    }
  }
  return 1;
}

IplImage *createRandomImage(int w, int h) {
  IplImage *img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
  int step = img->widthStep;
  int channels = img->nChannels;
  unsigned char *data = (unsigned char *)img->imageData;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      unsigned char *p = data + y * step + x * channels;
      p[0] = (unsigned char)(rand() % 256);
      p[1] = (unsigned char)(rand() % 256);
      p[2] = (unsigned char)(rand() % 256);
    }
  }
  return img;
}

int randomOddSize(int maxSize) {
  int halfCount = (maxSize + 1) / 2;
  return 2 * (rand() % halfCount) + 1;
}

int randomImageDim(void) {
  int r = rand() % 10;
  if (r == 0)
    return 1;
  if (r == 1)
    return 2;
  if (r == 2)
    return 3;
  return 4 + rand() % 40;
}

Filter randomConvexFilter(int w, int h) {
  double *data = malloc(w * h * sizeof(double));
  double sum = 0.0;
  for (int i = 0; i < w * h; i++) {
    data[i] = (double)rand() / RAND_MAX;
    sum += data[i];
  }
  if (sum == 0.0) {
    data[0] = 1.0;
    sum = 1.0;
  }
  for (int i = 0; i < w * h; i++) {
    data[i] /= sum;
  }

  Filter f = filter_create(w, h, data, 1.0, 0.0);
  free(data);
  return f;
}

Filter composeKernels(const Filter *a, const Filter *b) {
  int cw = a->width + b->width - 1;
  int ch = a->height + b->height - 1;
  double *data = calloc(cw * ch, sizeof(double));

  for (int ay = 0; ay < a->height; ay++)
    for (int ax = 0; ax < a->width; ax++)
      for (int by = 0; by < b->height; by++)
        for (int bx = 0; bx < b->width; bx++)
          data[(ay + by) * cw + (ax + bx)] +=
              a->matrix[ay][ax] * b->matrix[by][bx];

  Filter c = filter_create(cw, ch, data, 1.0, 0.0);
  free(data);
  return c;
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

  for (int y = 0; y < src->height; y++) {
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
  }

  cvReleaseMat(&kernel);
  cvReleaseImage(&padded);
  cvReleaseImage(&filtered64);
  return result;
}
