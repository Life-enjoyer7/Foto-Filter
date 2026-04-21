#include "filter.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// ========== Конструктор и деструктор ==========

Filter filter_create(int w, int h, double *data, double f, double b)
{
    Filter flt;
    flt.width = w;
    flt.height = h;
    flt.factor = f;
    flt.bias = b;

    flt.matrix = (double **)malloc(h * sizeof(double *));
    for (int i = 0; i < h; i++)
    {
        flt.matrix[i] = (double *)malloc(w * sizeof(double));
        for (int j = 0; j < w; j++)
        {
            flt.matrix[i][j] = data[i * w + j];
        }
    }

    return flt;
}

void filter_free(Filter *f)
{
    if (f->matrix)
    {
        for (int i = 0; i < f->height; i++)
        {
            free(f->matrix[i]);
        }
        free(f->matrix);
        f->matrix = NULL;
    }
}

// ========== Фильтры ==========

Filter filter_identity(void)
{
    double kernel[3][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_blur3x3(void)
{
    double kernel[3][3] = {{0.0, 0.2, 0.0}, {0.2, 0.2, 0.2}, {0.0, 0.2, 0.0}};
    return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_blur5x5(void)
{
    double kernel[5][5] = {
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0}};
    return filter_create(5, 5, &kernel[0][0], 1.0 / 13.0, 0.0);
}

Filter filter_gaussian3x3(void)
{
    double kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    return filter_create(3, 3, &kernel[0][0], 1.0 / 16.0, 0.0);
}

Filter filter_gaussian5x5(void)
{
    double kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}};
    return filter_create(5, 5, &kernel[0][0], 1.0 / 256.0, 0.0);
}

Filter filter_motionblur(void)
{
    double kernel[9][9] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 1}};
    return filter_create(9, 9, &kernel[0][0], 1.0 / 9.0, 0.0);
}

Filter filter_findedges1(void)
{
    double kernel[5][5] = {
        {0, 0, -1, 0, 0},
        {0, 0, -1, 0, 0},
        {0, 0, 2, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}};
    return filter_create(5, 5, &kernel[0][0], 1.0, 0.0);
}

Filter filter_findedges2(void)
{
    double kernel[5][5] = {
        {0, 0, -1, 0, 0},
        {0, 0, -1, 0, 0},
        {0, 0, 4, 0, 0},
        {0, 0, -1, 0, 0},
        {0, 0, -1, 0, 0}};
    return filter_create(5, 5, &kernel[0][0], 1.0, 0.0);
}

Filter filter_findedges3(void)
{
    double kernel[5][5] = {
        {-1, 0, 0, 0, 0},
        {0, -2, 0, 0, 0},
        {0, 0, 6, 0, 0},
        {0, 0, 0, -2, 0},
        {0, 0, 0, 0, -1}};
    return filter_create(5, 5, &kernel[0][0], 1.0, 0.0);
}

Filter filter_findedges4(void)
{
    double kernel[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_sharpen1(void)
{
    double kernel[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};
    return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_sharpen2(void)
{
    double kernel[5][5] = {
        {-1, -1, -1, -1, -1},
        {-1, 2, 2, 2, -1},
        {-1, 2, 8, 2, -1},
        {-1, 2, 2, 2, -1},
        {-1, -1, -1, -1, -1}};
    return filter_create(5, 5, &kernel[0][0], 1.0 / 8.0, 0.0);
}

Filter filter_sharpen3(void)
{
    double kernel[3][3] = {{1, 1, 1}, {1, -7, 1}, {1, 1, 1}};
    return filter_create(3, 3, &kernel[0][0], 1.0 / 8.0, 0.0);
}

Filter filter_emboss1(void)
{
    double kernel[3][3] = {{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
    return filter_create(3, 3, &kernel[0][0], 1.0, 128.0);
}

Filter filter_emboss2(void)
{
    double kernel[5][5] = {
        {-1, -1, -1, -1, 0},
        {-1, -1, -1, 0, 1},
        {-1, -1, 0, 1, 1},
        {-1, 0, 1, 1, 1},
        {0, 1, 1, 1, 1}};
    return filter_create(5, 5, &kernel[0][0], 1.0, 128.0);
}

// ========== Применение фильтра ==========

void applyFilter(const IplImage *src, IplImage *dst, const Filter *f)
{
    cvZero(dst);
    int w = src->width;
    int h = src->height;
    int step = src->widthStep;     // байт на строку
    int channels = src->nChannels; // обычно 3

    const unsigned char *src_data = (const unsigned char *)src->imageData;
    unsigned char *dst_data = (unsigned char *)dst->imageData;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            double red = 0.0, green = 0.0, blue = 0.0;

            for (int fy = 0; fy < f->height; fy++)
            {
                for (int fx = 0; fx < f->width; fx++)
                {
                    int ix = (x - f->width / 2 + fx + w) % w;
                    int iy = (y - f->height / 2 + fy + h) % h;

                    // Прямой доступ к пикселю
                    const unsigned char *pixel = src_data + iy * step + ix * channels;
                    blue += pixel[0] * f->matrix[fy][fx];
                    green += pixel[1] * f->matrix[fy][fx];
                    red += pixel[2] * f->matrix[fy][fx];
                }
            }

            int r = (int)(f->factor * red + f->bias);
            int g = (int)(f->factor * green + f->bias);
            int b = (int)(f->factor * blue + f->bias);

            r = MAX(0, MIN(255, r));
            g = MAX(0, MIN(255, g));
            b = MAX(0, MIN(255, b));

            unsigned char *out = dst_data + y * step + x * channels;
            out[0] = (unsigned char)b;
            out[1] = (unsigned char)g;
            out[2] = (unsigned char)r;
        }
    }
}