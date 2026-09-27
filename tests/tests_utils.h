#ifndef TESTS_UTILS_H
#define TESTS_UTILS_H

#include "../src/filter.h"
#include <stdio.h>

extern int g_test_failures;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      g_test_failures++;                                                       \
      printf("  FAILED: %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
    }                                                                          \
  } while (0)

#define RUN_TEST(fn, name)                                                     \
  do {                                                                         \
    int _failures_before = g_test_failures;                                    \
    fn();                                                                      \
    printf(                                                                    \
        "\n                                        %s %s (%d failure(s))\n",   \
        (name), g_test_failures == _failures_before ? "PASSED" : "FAILED",     \
        g_test_failures - _failures_before);                                   \
  } while (0)

extern const char *imagePaths[];
extern const char *imageNames[];

double timeFilterMs(const IplImage *src, IplImage *dst, const Filter *f);
double timeTwoFiltersMs(const IplImage *srcA, IplImage *dstA, const Filter *fA,
                        const IplImage *srcB, IplImage *dstB, const Filter *fB);

int imagesEqual(const IplImage *a, const IplImage *b);
int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance,
                      int *max_diff_out);

IplImage *createRandomImage(int w, int h);
int randomOddSize(int maxSize);
int randomImageDim(void);
Filter randomConvexFilter(int w, int h);
Filter composeKernels(const Filter *a, const Filter *b);

IplImage *referenceApplyFilter(const IplImage *src, const Filter *f);

#endif
