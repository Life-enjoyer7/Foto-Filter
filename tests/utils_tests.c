#include <stdlib.h>
// C++ Mat API - нужен только для referenceApplyFilter (сверка с OpenCV как эталоном).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "utils_tests.h"

const char *strategy_names[7] = {
    "Sequential",    // 0
    "Pixelwise",     // 1
    "By Rows",       // 2
    "By Cols",       // 3
    "Blocks 32x32",  // 4
    "Blocks 64x64",  // 5
    "Blocks 128x128" // 6
};

Filter (*const filter_factories[27])(void) = {
    filter_blur3x3,
    filter_blur5x5,
    filter_gaussian3x3,
    filter_gaussian5x5,
    filter_motionblur,
    filter_findedges1,
    filter_findedges2,
    filter_findedges3,
    filter_findedges4,
    filter_sharpen1,
    filter_sharpen2,
    filter_sharpen3,
    filter_emboss1,
    filter_emboss2,
    filter_identity,
    filter_shift_right,
    filter_shift_left,
    filter_shift_up,
    filter_shift_down,
    filter_shift_diag_up,
    filter_shift_diag_down,
    filter_blur3x3_padded,
    filter_gaussian3x3_padded,
    filter_findedges1_padded,
    filter_sharpen1_padded,
    filter_emboss1_padded,
    filter_zero,
};

IplImage *createRandomImage(int w, int h)
{
    IplImage *img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
    int step = img->widthStep;
    int channels = img->nChannels;
    unsigned char *data = (unsigned char *)img->imageData;

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            unsigned char *p = data + y * step + x * channels;
            p[0] = (unsigned char)(rand() % 256);
            p[1] = (unsigned char)(rand() % 256);
            p[2] = (unsigned char)(rand() % 256);
        }
    return img;
}

int randomImageDim(void)
{
    int r = rand() % 10;
    if (r == 0)
        return 1;
    if (r == 1)
        return 2;
    if (r == 2)
        return 3;
    return 4 + rand() % 57; // до 60
}

int randomFilterIdForDim(int minDim)
{
    static const int f3x3[] = {0, 2, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 26}; // half=1
    static const int f5x5[] = {1, 3, 5, 6, 7, 10, 13, 21, 22, 24, 25};              // half=2
    static const int f7x7[] = {23};                                                // half=3
    static const int f9x9[] = {4};                                                 // half=4

    int pool[27];
    int n = 0;
    for (int i = 0; i < 14; i++)
        pool[n++] = f3x3[i];
    if (minDim >= 2)
        for (int i = 0; i < 11; i++)
            pool[n++] = f5x5[i];
    if (minDim >= 3)
        pool[n++] = f7x7[0];
    if (minDim >= 4)
        pool[n++] = f9x9[0];

    return pool[rand() % n];
}

// cv::filter2D сам не поддерживает BORDER_WRAP ("BORDER_WRAP is not supported"
// в документации OpenCV), поэтому картинка вручную дополняется по кругу через
// cv::copyMakeBorder(..., BORDER_WRAP) на половину размера фильтра с каждой
// стороны, после чего вырезается центральная область того же размера, что и
// исходное изображение. Итоговый clamp+truncate делаем сами по double, а не
// через встроенную конвертацию OpenCV в 8U - у неё round(), а у applyFilter -
// truncate().
IplImage *referenceApplyFilter(const IplImage *src, const Filter *f)
{
    cv::Mat srcMat = cv::cvarrToMat(src);

    cv::Mat kernel(f->height, f->width, CV_64F);
    for (int y = 0; y < f->height; y++)
        for (int x = 0; x < f->width; x++)
            kernel.at<double>(y, x) = f->matrix[y][x] * f->factor;

    int padW = f->width / 2;
    int padH = f->height / 2;

    cv::Mat padded;
    cv::copyMakeBorder(srcMat, padded, padH, padH, padW, padW, cv::BORDER_WRAP);

    cv::Mat filtered64;
    cv::filter2D(padded, filtered64, CV_64F, kernel, cv::Point(-1, -1), f->bias, cv::BORDER_CONSTANT);

    cv::Mat cropped = filtered64(cv::Rect(padW, padH, src->width, src->height));

    IplImage *result = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
    int step = result->widthStep;
    int channels = result->nChannels;
    unsigned char *dst_data = (unsigned char *)result->imageData;

    for (int y = 0; y < src->height; y++)
    {
        for (int x = 0; x < src->width; x++)
        {
            const cv::Vec3d &px = cropped.at<cv::Vec3d>(y, x);
            unsigned char *out = dst_data + y * step + x * channels;
            for (int c = 0; c < 3; c++)
            {
                int v = (int)px[c];
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                out[c] = (unsigned char)v;
            }
        }
    }

    return result;
}

int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance)
{
    if (a->width != b->width || a->height != b->height || a->nChannels != b->nChannels)
        return 0;

    int step = a->widthStep;
    int channels = a->nChannels;
    const unsigned char *da = (const unsigned char *)a->imageData;
    const unsigned char *db = (const unsigned char *)b->imageData;

    for (int y = 0; y < a->height; y++)
        for (int x = 0; x < a->width; x++)
        {
            const unsigned char *pa = da + y * step + x * channels;
            const unsigned char *pb = db + y * step + x * channels;
            for (int c = 0; c < 3; c++)
                if (abs((int)pa[c] - (int)pb[c]) > tolerance)
                    return 0;
        }
    return 1;
}
