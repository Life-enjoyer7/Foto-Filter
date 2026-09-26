#ifndef MAIN_UTILS_H
#define MAIN_UTILS_H

#include "filter.h"



double get_time_ms(void);



void benchmark_filter(FilterFn fn, const IplImage *src, IplImage *dst, const Filter *f, int repeat,
                       double *out_min, double *out_mean, double *out_median);

#endif
