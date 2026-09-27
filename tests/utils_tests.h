#ifndef UTILS_TESTS_H
#define UTILS_TESTS_H

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

extern const char *strategy_names[7];

#define PIPELINE_BENCH_REPEAT 2

#define MAX_TRIAL_IMAGES 5
#define RANDOM_TEST_TRIALS 30

IplImage *createRandomImage(int w, int h);

int randomImageDim(void);

int randomFilterIdForDim(int minDim);

IplImage *referenceApplyFilter(const IplImage *src, const Filter *f);

int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance);

#endif
