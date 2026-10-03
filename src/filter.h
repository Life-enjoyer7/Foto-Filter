#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core/core_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc/imgproc_c.h>

typedef struct {
  int width;
  int height;
  double **matrix;
  double factor;
  double bias;
} Filter;

Filter filter_create(int w, int h, const double *data, double f, double b);
void filter_free(Filter *f);

Filter filter_identity(void);
Filter filter_blur3x3(void);
Filter filter_blur5x5(void);
Filter filter_gaussian3x3(void);
Filter filter_gaussian5x5(void);
Filter filter_motionblur(void);
Filter filter_findedges1(void);
Filter filter_findedges2(void);
Filter filter_findedges3(void);
Filter filter_findedges4(void);
Filter filter_sharpen1(void);
Filter filter_sharpen2(void);
Filter filter_sharpen3(void);
Filter filter_emboss1(void);
Filter filter_emboss2(void);

#define NUM_FILTERS 15
Filter filter_by_id(int id);
const char *filter_name(int id);

typedef void (*FilterFn)(const IplImage *, IplImage *, const Filter *);

typedef struct {
  const char *name;
  FilterFn fn;
} NamedStrategy;

#define NUM_STRATEGIES 7
extern const NamedStrategy allStrategies[NUM_STRATEGIES];

void applyFilter(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelPixelwise(const IplImage *src, IplImage *dst,
                                  const Filter *f);
void applyFilterParallelByRows(const IplImage *src, IplImage *dst,
                               const Filter *f);
void applyFilterParallelByCols(const IplImage *src, IplImage *dst,
                               const Filter *f);
void applyFilterParallelByBlocks32(const IplImage *src, IplImage *dst,
                                   const Filter *f);
void applyFilterParallelByBlocks64(const IplImage *src, IplImage *dst,
                                   const Filter *f);
void applyFilterParallelByBlocks128(const IplImage *src, IplImage *dst,
                                    const Filter *f);

void set_conv_threads(int n);

#endif