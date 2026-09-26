#include <stdlib.h>
#include <time.h>
#include "main_utils.h"

double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

void benchmark_filter(FilterFn fn, const IplImage *src, IplImage *dst, const Filter *f, int repeat,
                       double *out_min, double *out_mean, double *out_median)
{
    fn(src, dst, f);

    double *samples = malloc(repeat * sizeof(double));
    double sum = 0.0;
    for (int i = 0; i < repeat; i++)
    {
        double start = get_time_ms();
        fn(src, dst, f);
        double end = get_time_ms();
        samples[i] = end - start;
        sum += samples[i];
    }

    qsort(samples, repeat, sizeof(double), compare_double);

    *out_min = samples[0];
    *out_mean = sum / repeat;
    *out_median = (repeat % 2 == 0)
                      ? (samples[repeat / 2 - 1] + samples[repeat / 2]) / 2.0
                      : samples[repeat / 2];

    free(samples);
}
