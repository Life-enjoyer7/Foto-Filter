#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include "../src/filter.h"

const char *imagePaths[15] = {
    "images/bugatti_1200x600.jpg",
    "images/bugatti_1536x2048.jpg",
    "images/bugatti_3275x4096.jpg",
    "images/ferari_320x320.jpg",
    "images/ferrari_2560x1440.jpg",
    "images/ford_1080x1080.jpg",
    "images/lambo_236x236.jpg",
    "images/lambo_1080x1349.jpg",
    "images/maseratti_2048x2048.jpg",
    "images/mustang_736x736.jpg",
    "images/sportcar_474x503.jpg",
    "images/sportcar_736x981.jpg",
    "images/sportcar_3823x4237.jpg",
    "images/bugatti_big_size.jpg",
    "images/car_1200x687.jpeg"};

const char *imageNames[15] = {
    "bugatti_1200x600.jpg",
    "bugatti_1536x2048.jpg",
    "bugatti_3275x4096.jpg",
    "ferari_320x320.jpg",
    "ferrari_2560x1440.jpg",
    "ford_1080x1080.jpg",
    "lambo_236x236.jpg",
    "lambo_1080x1349.jpg",
    "maseratti_2048x2048.jpg",
    "mustang_736x736.jpg",
    "sportcar_474x503.jpg",
    "sportcar_736x981.jpg",
    "sportcar_3823x4237.jpg",
    "bugatti_big_size.jpg",
    "car_1200x687.jpeg"};

double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int imagesEqual(const IplImage *a, const IplImage *b)
{
    if (a->width != b->width || a->height != b->height || a->nChannels != b->nChannels)
        return 0;

    int step = a->widthStep;
    int channels = a->nChannels;
    const unsigned char *data_a = (const unsigned char *)a->imageData;
    const unsigned char *data_b = (const unsigned char *)b->imageData;

    for (int y = 0; y < a->height; y++)
    {
        for (int x = 0; x < a->width; x++)
        {
            const unsigned char *pa = data_a + y * step + x * channels;
            const unsigned char *pb = data_b + y * step + x * channels;
            if (pa[0] != pb[0] || pa[1] != pb[1] || pa[2] != pb[2])
                return 0;
        }
    }
    return 1;
}

void testIdentityFilter(void)
{
    printf("\n");
    printf("                                        TEST 1: IDENTITY FILTER                         \n");

    Filter identity = filter_identity();
    double times[15] = {0};
    int valid_count = 0;

    for (int j = 0; j < 15; j++)
    {
        IplImage *img = cvLoadImage(imagePaths[j], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %s\n", imagePaths[j]);
            continue;
        }

        IplImage *result = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilter(img, result, &identity);
        double end = get_time_ms();
        times[valid_count] = end - start;

        // Проверка: identity фильтр не должен изменять изображение
        int eq = imagesEqual(img, result);
        if (!eq)
        {
            printf("Results do not match original for image %d!\n", j);
        }
        assert(eq);

        cvReleaseImage(&result);
        cvReleaseImage(&img);
        valid_count++;
    }

    printf("\n                                        PERFORMANCE RESULTS\n\n");
    double total = 0;
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[i], imageNames[i]);
        total += times[i];
    }
    printf("\n  Total: %8.2f ms\n", total);
    printf("\n                                        TEST 1 PASSED\n");
    filter_free(&identity);
}

void testShiftComposition(void)
{
    printf("\n");
    printf("                                        TEST 2: SHIFT COMPOSITION                         \n");

    double times[3][15] = {0}; // 3 композиции: RightLeft, UpDown, Diag
    int valid_count = 0;

    double kernel_right[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}};
    double kernel_left[3][3] = {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    double kernel_up[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 1, 0}};
    double kernel_down[3][3] = {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}};
    double kernel_diag_up[3][3] = {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}};
    double kernel_diag_down[3][3] = {{0, 0, 1}, {0, 0, 0}, {0, 0, 0}};

    Filter shiftRight = filter_create(3, 3, &kernel_right[0][0], 1.0, 0.0);
    Filter shiftLeft = filter_create(3, 3, &kernel_left[0][0], 1.0, 0.0);
    Filter shiftUp = filter_create(3, 3, &kernel_up[0][0], 1.0, 0.0);
    Filter shiftDown = filter_create(3, 3, &kernel_down[0][0], 1.0, 0.0);
    Filter shiftDiagUp = filter_create(3, 3, &kernel_diag_up[0][0], 1.0, 0.0);
    Filter shiftDiagDown = filter_create(3, 3, &kernel_diag_down[0][0], 1.0, 0.0);

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        // Right-Left
        IplImage *temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilter(img, temp, &shiftRight);
        applyFilter(temp, final, &shiftLeft);
        double end = get_time_ms();
        times[0][valid_count] = end - start;
        int eq = imagesEqual(img, final);
        if (!eq)
            printf("Right-Left composition failed for image %d\n", i);
        assert(eq);
        cvReleaseImage(&temp);
        cvReleaseImage(&final);

        // Up-Down
        temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilter(img, temp, &shiftUp);
        applyFilter(temp, final, &shiftDown);
        end = get_time_ms();
        times[1][valid_count] = end - start;
        eq = imagesEqual(img, final);
        if (!eq)
            printf("Up-Down composition failed for image %d\n", i);
        assert(eq);
        cvReleaseImage(&temp);
        cvReleaseImage(&final);

        // Diag
        temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilter(img, temp, &shiftDiagUp);
        applyFilter(temp, final, &shiftDiagDown);
        end = get_time_ms();
        times[2][valid_count] = end - start;
        eq = imagesEqual(img, final);
        if (!eq)
            printf("Diag composition failed for image %d\n", i);
        assert(eq);
        cvReleaseImage(&temp);
        cvReleaseImage(&final);

        cvReleaseImage(&img);
        valid_count++;
    }

    filter_free(&shiftRight);
    filter_free(&shiftLeft);
    filter_free(&shiftUp);
    filter_free(&shiftDown);
    filter_free(&shiftDiagUp);
    filter_free(&shiftDiagDown);

    printf("\n                                        SHIFT COMPOSITION RESULTS\n\n");
    printf("LEFT-RIGHT:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[0][i], imageNames[i]);
    }
    printf("\nUP-DOWN:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[1][i], imageNames[i]);
    }
    printf("\nDIAG:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[2][i], imageNames[i]);
    }

    printf("\n                                        TEST 2 PASSED\n");
}

void testZeroPadding(void)
{
    printf("\n");
    printf("                                        TEST 3: ZERO PADDING                                         \n");

    double times[5][15] = {0}; // 5 фильтров
    int valid_count = 0;

    Filter original_filters[5];
    original_filters[0] = filter_blur3x3();
    original_filters[1] = filter_gaussian3x3();
    original_filters[2] = filter_findedges1();
    original_filters[3] = filter_sharpen1();
    original_filters[4] = filter_emboss1();

    double kernel_blur_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0.2, 0, 0},
        {0, 0.2, 0.2, 0.2, 0},
        {0, 0, 0.2, 0, 0},
        {0, 0, 0, 0, 0}};

    double kernel_gauss_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 1, 2, 1, 0},
        {0, 2, 4, 2, 0},
        {0, 1, 2, 1, 0},
        {0, 0, 0, 0, 0}};

    double kernel_edges_padded[7][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, -1, 0, 0, 0},
        {0, 0, 0, -1, 0, 0, 0},
        {0, 0, 0, 2, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0}};

    double kernel_sharpen_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, -1, -1, -1, 0},
        {0, -1, 9, -1, 0},
        {0, -1, -1, -1, 0},
        {0, 0, 0, 0, 0}};

    double kernel_emboss_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, -1, -1, 0, 0},
        {0, -1, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {0, 0, 0, 0, 0}};

    Filter padded_filters[5];
    padded_filters[0] = filter_create(5, 5, &kernel_blur_padded[0][0], 1.0, 0.0);
    padded_filters[1] = filter_create(5, 5, &kernel_gauss_padded[0][0], 1.0 / 16.0, 0.0);
    padded_filters[2] = filter_create(7, 7, &kernel_edges_padded[0][0], 1.0, 0.0);
    padded_filters[3] = filter_create(5, 5, &kernel_sharpen_padded[0][0], 1.0, 0.0);
    padded_filters[4] = filter_create(5, 5, &kernel_emboss_padded[0][0], 1.0, 128.0);

    const char *filter_names[5] = {"blur3x3", "gaussian3x3", "findedges1", "sharpen1", "emboss1"};

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        for (int j = 0; j < 5; j++)
        {
            IplImage *result_orig = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            IplImage *result_padded = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

            double start = get_time_ms();
            applyFilter(img, result_orig, &original_filters[j]);
            applyFilter(img, result_padded, &padded_filters[j]);
            double end = get_time_ms();
            times[j][valid_count] = end - start;

            int eq = imagesEqual(result_orig, result_padded);
            if (!eq)
            {
                printf("ERROR: Filter %s on image %d: results do not match!\n", filter_names[j], i);
            }
            assert(eq);

            cvReleaseImage(&result_orig);
            cvReleaseImage(&result_padded);
        }

        cvReleaseImage(&img);
        valid_count++;
    }

    for (int j = 0; j < 5; j++)
    {
        filter_free(&padded_filters[j]);
        filter_free(&original_filters[j]);
    }

    printf("\n                                        ZERO PADDING RESULTS\n\n");
    for (int j = 0; j < 5; j++)
    {
        printf("%s:\n", filter_names[j]);
        for (int i = 0; i < valid_count; i++)
        {
            printf("  %8.2f ms  for  %s\n", times[j][i], imageNames[i]);
        }
        printf("\n");
    }

    printf("\n                                        TEST 3 PASSED\n");
}

void testZeroFilter(void)
{
    printf("\n");
    printf("                                        TEST 4: ZERO FILTER                                         \n");

    double times[15] = {0};
    int valid_count = 0;

    double kernel_zero[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Filter zero_filter = filter_create(3, 3, &kernel_zero[0][0], 1.0, 0.0);

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        IplImage *result = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilter(img, result, &zero_filter);
        double end = get_time_ms();
        times[valid_count] = end - start;

        // Проверка: все пиксели должны быть чёрными (0,0,0)
        int step = result->widthStep;
        int channels = result->nChannels;
        const unsigned char *data = (const unsigned char *)result->imageData;
        int all_black = 1;
        for (int y = 0; y < result->height && all_black; y++)
        {
            for (int x = 0; x < result->width; x++)
            {
                const unsigned char *pixel = data + y * step + x * channels;
                if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                {
                    all_black = 0;
                    break;
                }
            }
        }
        if (!all_black)
        {
            printf("ERROR: Zero filter on image %d: result is not all black!\n", i);
        }
        assert(all_black);

        cvReleaseImage(&result);
        cvReleaseImage(&img);
        valid_count++;
    }

    filter_free(&zero_filter);

    printf("\n                                        ZERO FILTER RESULTS\n\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[i], imageNames[i]);
    }

    printf("\n                                        TEST 4 PASSED\n");
}

int main(void)
{
    testIdentityFilter();
    testShiftComposition();
    testZeroPadding();
    testZeroFilter();
    return 0;
}