#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

typedef struct
{
    int width;
    int height;
    double **matrix;
    double factor;
    double bias;
} Filter;

// Конструктор и деструктор
Filter filter_create(int w, int h, double *data, double f, double b);
void filter_free(Filter *f);

// Фильтры
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

// Применение фильтра
void applyFilter(const IplImage *src, IplImage *dst, const Filter *f);

#endif