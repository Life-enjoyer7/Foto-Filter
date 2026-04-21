#include "filter.h"
#include <pthread.h>
#include <unistd.h>
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

void applyFilter(const IplImage *src, IplImage *dst, const Filter *f)
{
    cvZero(dst);
    int w = src->width;
    int h = src->height;
    int step = src->widthStep;
    /
        int channels = src->nChannels;

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

typedef struct
{
    const IplImage *src;
    IplImage *dst;
    const Filter *f;
    int w;
    int h;
    int startIdx;
    int endIdx;
} ThreadArgs;

static void *processPixelRange(void *args)
{
    ThreadArgs *a = (ThreadArgs *)args;
    const IplImage *src = a->src;
    IplImage *dst = a->dst;
    const Filter *f = a->f;
    int w = a->w;
    int h = a->h;
    int startIdx = a->startIdx;
    int endIdx = a->endIdx;

    int step = src->widthStep;
    int channels = src->nChannels;
    const unsigned char *src_data = (const unsigned char *)src->imageData;
    unsigned char *dst_data = (unsigned char *)dst->imageData;

    for (int idx = startIdx; idx < endIdx; idx++)
    {
        int y = idx / w;
        int x = idx % w;

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

        r = (r < 0) ? 0 : (r > 255 ? 255 : r);
        g = (g < 0) ? 0 : (g > 255 ? 255 : g);
        b = (b < 0) ? 0 : (b > 255 ? 255 : b);

        unsigned char *out = dst_data + y * step + x * channels;
        out[0] = (unsigned char)b;
        out[1] = (unsigned char)g;
        out[2] = (unsigned char)r;
    }

    return NULL;
}

void applyFilterParallelPixelwise(const IplImage *src, IplImage *dst, const Filter *f)
{

    cvZero(dst);

    int w = src->width;
    int h = src->height;
    int totalPixels = w * h;

    int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (numThreads <= 0)
        numThreads = 4;

    int pixelsPerThread = totalPixels / numThreads;
    int remainder = totalPixels % numThreads;

    pthread_t threads[numThreads];
    ThreadArgs args[numThreads];

    for (int t = 0; t < numThreads; t++)
    {
        int startIdx = t * pixelsPerThread + (t < remainder ? t : remainder);
        int endIdx = startIdx + pixelsPerThread + (t < remainder ? 1 : 0);

        args[t].src = src;
        args[t].dst = dst;
        args[t].f = f;
        args[t].w = w;
        args[t].h = h;
        args[t].startIdx = startIdx;
        args[t].endIdx = endIdx;

        pthread_create(&threads[t], NULL, processPixelRange, &args[t]);
    }

    for (int t = 0; t < numThreads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}

typedef struct
{
    const IplImage *src;
    IplImage *dst;
    const Filter *f;
    int w;
    int h;
    int startRow;
    int endRow;
} ThreadArgsRows;

static void *processRowRange(void *args)
{
    ThreadArgsRows *a = (ThreadArgsRows *)args;
    const IplImage *src = a->src;
    IplImage *dst = a->dst;
    const Filter *f = a->f;
    int w = a->w;
    int h = a->h;
    int startRow = a->startRow;
    int endRow = a->endRow;

    int step = src->widthStep;
    int channels = src->nChannels;
    const unsigned char *src_data = (const unsigned char *)src->imageData;
    unsigned char *dst_data = (unsigned char *)dst->imageData;

    for (int y = startRow; y < endRow; y++)
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

                    const unsigned char *pixel = src_data + iy * step + ix * channels;
                    blue += pixel[0] * f->matrix[fy][fx];
                    green += pixel[1] * f->matrix[fy][fx];
                    red += pixel[2] * f->matrix[fy][fx];
                }
            }

            int r = (int)(f->factor * red + f->bias);
            int g = (int)(f->factor * green + f->bias);
            int b = (int)(f->factor * blue + f->bias);

            r = (r < 0) ? 0 : (r > 255 ? 255 : r);
            g = (g < 0) ? 0 : (g > 255 ? 255 : g);
            b = (b < 0) ? 0 : (b > 255 ? 255 : b);

            unsigned char *out = dst_data + y * step + x * channels;
            out[0] = (unsigned char)b;
            out[1] = (unsigned char)g;
            out[2] = (unsigned char)r;
        }
    }

    return NULL;
}

void applyFilterParallelByRows(const IplImage *src, IplImage *dst, const Filter *f)
{

    cvZero(dst);

    int w = src->width;
    int h = src->height;

    int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (numThreads <= 0)
        numThreads = 4;

    if (numThreads > h)
        numThreads = h;

    int rowsPerThread = h / numThreads;
    int remainder = h % numThreads;

    pthread_t threads[numThreads];
    ThreadArgsRows args[numThreads];

    for (int t = 0; t < numThreads; t++)
    {
        int startRow = t * rowsPerThread + (t < remainder ? t : remainder);
        int endRow = startRow + rowsPerThread + (t < remainder ? 1 : 0);

        args[t].src = src;
        args[t].dst = dst;
        args[t].f = f;
        args[t].w = w;
        args[t].h = h;
        args[t].startRow = startRow;
        args[t].endRow = endRow;

        pthread_create(&threads[t], NULL, processRowRange, &args[t]);
    }

    for (int t = 0; t < numThreads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}

typedef struct
{
    const IplImage *src;
    IplImage *dst;
    const Filter *f;
    int w;
    int h;
    int startCol;
    int endCol;
} ThreadArgsCols;

static void *processColRange(void *args)
{
    ThreadArgsCols *a = (ThreadArgsCols *)args;
    const IplImage *src = a->src;
    IplImage *dst = a->dst;
    const Filter *f = a->f;
    int w = a->w;
    int h = a->h;
    int startCol = a->startCol;
    int endCol = a->endCol;

    int step = src->widthStep;
    int channels = src->nChannels;
    const unsigned char *src_data = (const unsigned char *)src->imageData;
    unsigned char *dst_data = (unsigned char *)dst->imageData;

    for (int y = 0; y < h; y++)
    {
        for (int x = startCol; x < endCol; x++)
        {
            double red = 0.0, green = 0.0, blue = 0.0;

            for (int fy = 0; fy < f->height; fy++)
            {
                for (int fx = 0; fx < f->width; fx++)
                {
                    int ix = (x - f->width / 2 + fx + w) % w;
                    int iy = (y - f->height / 2 + fy + h) % h;

                    const unsigned char *pixel = src_data + iy * step + ix * channels;
                    blue += pixel[0] * f->matrix[fy][fx];
                    green += pixel[1] * f->matrix[fy][fx];
                    red += pixel[2] * f->matrix[fy][fx];
                }
            }

            int r = (int)(f->factor * red + f->bias);
            int g = (int)(f->factor * green + f->bias);
            int b = (int)(f->factor * blue + f->bias);

            r = (r < 0) ? 0 : (r > 255 ? 255 : r);
            g = (g < 0) ? 0 : (g > 255 ? 255 : g);
            b = (b < 0) ? 0 : (b > 255 ? 255 : b);

            unsigned char *out = dst_data + y * step + x * channels;
            out[0] = (unsigned char)b;
            out[1] = (unsigned char)g;
            out[2] = (unsigned char)r;
        }
    }

    return NULL;
}

void applyFilterParallelByCols(const IplImage *src, IplImage *dst, const Filter *f)
{

    cvZero(dst);

    int w = src->width;
    int h = src->height;

    int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (numThreads <= 0)
        numThreads = 4;

    if (numThreads > w)
        numThreads = w;

    int colsPerThread = w / numThreads;
    int remainder = w % numThreads;

    pthread_t threads[numThreads];
    ThreadArgsCols args[numThreads];

    for (int t = 0; t < numThreads; t++)
    {
        int startCol = t * colsPerThread + (t < remainder ? t : remainder);
        int endCol = startCol + colsPerThread + (t < remainder ? 1 : 0);

        args[t].src = src;
        args[t].dst = dst;
        args[t].f = f;
        args[t].w = w;
        args[t].h = h;
        args[t].startCol = startCol;
        args[t].endCol = endCol;

        pthread_create(&threads[t], NULL, processColRange, &args[t]);
    }

    for (int t = 0; t < numThreads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}