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

// Фильтры сдвига
Filter filter_shift_right(void);
Filter filter_shift_left(void);
Filter filter_shift_up(void);
Filter filter_shift_down(void);
Filter filter_shift_diag_up(void);
Filter filter_shift_diag_down(void);

// Padded фильтры
Filter filter_blur3x3_padded(void);
Filter filter_gaussian3x3_padded(void);
Filter filter_findedges1_padded(void);
Filter filter_sharpen1_padded(void);
Filter filter_emboss1_padded(void);

// Нулевой фильтр
Filter filter_zero(void);

// Применение фильтра (все версии)
void applyFilter(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelPixelwise(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelByRows(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelByCols(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelByBlocks(const IplImage *src, IplImage *dst, const Filter *f, int blockW, int blockH);
void applyFilterParallelByBlocks32(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelByBlocks64(const IplImage *src, IplImage *dst, const Filter *f);
void applyFilterParallelByBlocks128(const IplImage *src, IplImage *dst, const Filter *f);

#endif