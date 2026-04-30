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

    double times_pixel[15] = {0};
    double times_rows[15] = {0};
    double times_cols[15] = {0};
    double times_blocks32[15] = {0};
    double times_blocks64[15] = {0};
    double times_blocks128[15] = {0};

    int valid_count = 0;

    for (int j = 0; j < 15; j++)
    {
        IplImage *img = cvLoadImage(imagePaths[j], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image\n");
            continue;
        }

        // Параллельная попиксельно
        IplImage *result_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilterParallelPixelwise(img, result_pixel, &identity);
        double end = get_time_ms();
        times_pixel[valid_count] = end - start;

        // Параллельная по строкам
        IplImage *result_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, result_rows, &identity);
        end = get_time_ms();
        times_rows[valid_count] = end - start;

        // Параллельная по столбцам
        IplImage *result_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, result_cols, &identity);
        end = get_time_ms();
        times_cols[valid_count] = end - start;

        // Параллельная по блокам 32x32
        IplImage *result_blocks32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks32(img, result_blocks32, &identity);
        end = get_time_ms();
        times_blocks32[valid_count] = end - start;

        // Параллельная по блокам 64x64
        IplImage *result_blocks64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks64(img, result_blocks64, &identity);
        end = get_time_ms();
        times_blocks64[valid_count] = end - start;

        // Параллельная по блокам 128x128
        IplImage *result_blocks128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks128(img, result_blocks128, &identity);
        end = get_time_ms();
        times_blocks128[valid_count] = end - start;

        // Проверка: все параллельные версии должны давать одинаковый результат
        int eq_pixel_rows = imagesEqual(result_pixel, result_rows);
        int eq_pixel_cols = imagesEqual(result_pixel, result_cols);
        int eq_pixel_32 = imagesEqual(result_pixel, result_blocks32);
        int eq_pixel_64 = imagesEqual(result_pixel, result_blocks64);
        int eq_pixel_128 = imagesEqual(result_pixel, result_blocks128);

        if (!eq_pixel_rows || !eq_pixel_cols || !eq_pixel_32 || !eq_pixel_64 || !eq_pixel_128)
        {
            printf("Results do not match between parallel strategies!\n");
        }
        assert(eq_pixel_rows && eq_pixel_cols && eq_pixel_32 && eq_pixel_64 && eq_pixel_128);

        cvReleaseImage(&result_pixel);
        cvReleaseImage(&result_rows);
        cvReleaseImage(&result_cols);
        cvReleaseImage(&result_blocks32);
        cvReleaseImage(&result_blocks64);
        cvReleaseImage(&result_blocks128);
        cvReleaseImage(&img);

        valid_count++;
    }

    printf("\n");
    printf("                                        PERFORMANCE RESULTS (PARALLEL ONLY)                         \n");

    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for \t %-30s \n",
               times_pixel[i], times_rows[i], times_cols[i], times_blocks32[i], times_blocks64[i], times_blocks128[i], imageNames[i]);
    }

    double total_pixel = 0, total_rows = 0, total_cols = 0;
    double total_blocks32 = 0, total_blocks64 = 0, total_blocks128 = 0;
    for (int i = 0; i < valid_count; i++)
    {
        total_pixel += times_pixel[i];
        total_rows += times_rows[i];
        total_cols += times_cols[i];
        total_blocks32 += times_blocks32[i];
        total_blocks64 += times_blocks64[i];
        total_blocks128 += times_blocks128[i];
    }
    printf("\n");
    printf("Total: \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms \n",
           total_pixel, total_rows, total_cols, total_blocks32, total_blocks64, total_blocks128);

    printf("\n");
    printf("                                        TEST 1 PASSED                                  \n");
    filter_free(&identity);
}

void testShiftComposition(void)
{
    printf("\n");
    printf("                                        TEST 2: SHIFT COMPOSITION                         \n");

    double times[3][6][15] = {0}; // 3 композиции × 6 стратегий (без Seq)
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
        // Попиксельно
        IplImage *temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilterParallelPixelwise(img, temp_pixel, &shiftRight);
        applyFilterParallelPixelwise(temp_pixel, final_pixel, &shiftLeft);
        double end = get_time_ms();
        times[0][0][valid_count] = end - start;

        // По строкам
        IplImage *temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, temp_rows, &shiftRight);
        applyFilterParallelByRows(temp_rows, final_rows, &shiftLeft);
        end = get_time_ms();
        times[0][1][valid_count] = end - start;

        // По столбцам
        IplImage *temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, temp_cols, &shiftRight);
        applyFilterParallelByCols(temp_cols, final_cols, &shiftLeft);
        end = get_time_ms();
        times[0][2][valid_count] = end - start;

        // Блоки 32x32
        IplImage *temp_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks32(img, temp_32, &shiftRight);
        applyFilterParallelByBlocks32(temp_32, final_32, &shiftLeft);
        end = get_time_ms();
        times[0][3][valid_count] = end - start;

        // Блоки 64x64
        IplImage *temp_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks64(img, temp_64, &shiftRight);
        applyFilterParallelByBlocks64(temp_64, final_64, &shiftLeft);
        end = get_time_ms();
        times[0][4][valid_count] = end - start;

        // Блоки 128x128
        IplImage *temp_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks128(img, temp_128, &shiftRight);
        applyFilterParallelByBlocks128(temp_128, final_128, &shiftLeft);
        end = get_time_ms();
        times[0][5][valid_count] = end - start;

        int eq = imagesEqual(img, final_pixel);
        if (!eq)
            printf("Right-Left composition failed for image %d\n", i);
        assert(eq);

        cvReleaseImage(&temp_pixel);
        cvReleaseImage(&final_pixel);
        cvReleaseImage(&temp_rows);
        cvReleaseImage(&final_rows);
        cvReleaseImage(&temp_cols);
        cvReleaseImage(&final_cols);
        cvReleaseImage(&temp_32);
        cvReleaseImage(&final_32);
        cvReleaseImage(&temp_64);
        cvReleaseImage(&final_64);
        cvReleaseImage(&temp_128);
        cvReleaseImage(&final_128);

        // Up-Down (аналогично, только с shiftUp/shiftDown)
        temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, temp_pixel, &shiftUp);
        applyFilterParallelPixelwise(temp_pixel, final_pixel, &shiftDown);
        end = get_time_ms();
        times[1][0][valid_count] = end - start;

        temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, temp_rows, &shiftUp);
        applyFilterParallelByRows(temp_rows, final_rows, &shiftDown);
        end = get_time_ms();
        times[1][1][valid_count] = end - start;

        temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, temp_cols, &shiftUp);
        applyFilterParallelByCols(temp_cols, final_cols, &shiftDown);
        end = get_time_ms();
        times[1][2][valid_count] = end - start;

        temp_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks32(img, temp_32, &shiftUp);
        applyFilterParallelByBlocks32(temp_32, final_32, &shiftDown);
        end = get_time_ms();
        times[1][3][valid_count] = end - start;

        temp_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks64(img, temp_64, &shiftUp);
        applyFilterParallelByBlocks64(temp_64, final_64, &shiftDown);
        end = get_time_ms();
        times[1][4][valid_count] = end - start;

        temp_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks128(img, temp_128, &shiftUp);
        applyFilterParallelByBlocks128(temp_128, final_128, &shiftDown);
        end = get_time_ms();
        times[1][5][valid_count] = end - start;

        eq = imagesEqual(img, final_pixel);
        if (!eq)
            printf("Up-Down composition failed for image %d\n", i);
        assert(eq);

        cvReleaseImage(&temp_pixel);
        cvReleaseImage(&final_pixel);
        cvReleaseImage(&temp_rows);
        cvReleaseImage(&final_rows);
        cvReleaseImage(&temp_cols);
        cvReleaseImage(&final_cols);
        cvReleaseImage(&temp_32);
        cvReleaseImage(&final_32);
        cvReleaseImage(&temp_64);
        cvReleaseImage(&final_64);
        cvReleaseImage(&temp_128);
        cvReleaseImage(&final_128);

        // Diag (с shiftDiagUp/shiftDiagDown)
        temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, temp_pixel, &shiftDiagUp);
        applyFilterParallelPixelwise(temp_pixel, final_pixel, &shiftDiagDown);
        end = get_time_ms();
        times[2][0][valid_count] = end - start;

        temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, temp_rows, &shiftDiagUp);
        applyFilterParallelByRows(temp_rows, final_rows, &shiftDiagDown);
        end = get_time_ms();
        times[2][1][valid_count] = end - start;

        temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, temp_cols, &shiftDiagUp);
        applyFilterParallelByCols(temp_cols, final_cols, &shiftDiagDown);
        end = get_time_ms();
        times[2][2][valid_count] = end - start;

        temp_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks32(img, temp_32, &shiftDiagUp);
        applyFilterParallelByBlocks32(temp_32, final_32, &shiftDiagDown);
        end = get_time_ms();
        times[2][3][valid_count] = end - start;

        temp_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks64(img, temp_64, &shiftDiagUp);
        applyFilterParallelByBlocks64(temp_64, final_64, &shiftDiagDown);
        end = get_time_ms();
        times[2][4][valid_count] = end - start;

        temp_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks128(img, temp_128, &shiftDiagUp);
        applyFilterParallelByBlocks128(temp_128, final_128, &shiftDiagDown);
        end = get_time_ms();
        times[2][5][valid_count] = end - start;

        eq = imagesEqual(img, final_pixel);
        if (!eq)
            printf("Diag composition failed for image %d\n", i);
        assert(eq);

        cvReleaseImage(&temp_pixel);
        cvReleaseImage(&final_pixel);
        cvReleaseImage(&temp_rows);
        cvReleaseImage(&final_rows);
        cvReleaseImage(&temp_cols);
        cvReleaseImage(&final_cols);
        cvReleaseImage(&temp_32);
        cvReleaseImage(&final_32);
        cvReleaseImage(&temp_64);
        cvReleaseImage(&final_64);
        cvReleaseImage(&temp_128);
        cvReleaseImage(&final_128);

        cvReleaseImage(&img);
        valid_count++;
    }

    filter_free(&shiftRight);
    filter_free(&shiftLeft);
    filter_free(&shiftUp);
    filter_free(&shiftDown);
    filter_free(&shiftDiagUp);
    filter_free(&shiftDiagDown);

    printf("\n                                        SHIFT COMPOSITION RESULTS (PARALLEL ONLY)\n\n");
    printf("LEFT-RIGHT:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for \t %-30s \n",
               times[0][0][i], times[0][1][i], times[0][2][i], times[0][3][i], times[0][4][i], times[0][5][i], imageNames[i]);
    }
    printf("\nUP-DOWN:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for \t %-30s \n",
               times[1][0][i], times[1][1][i], times[1][2][i], times[1][3][i], times[1][4][i], times[1][5][i], imageNames[i]);
    }
    printf("\nDIAG:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for \t %-30s \n",
               times[2][0][i], times[2][1][i], times[2][2][i], times[2][3][i], times[2][4][i], times[2][5][i], imageNames[i]);
    }

    printf("\n                                        TEST 2 PASSED\n");
}

void testZeroPadding(void)
{
    printf("\n");
    printf("                                        TEST 3: ZERO PADDING                                         \n");
    printf("\n");

    double times[5][6][15] = {0}; // 5 фильтров × 6 стратегий × 15 изображений
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
            // Попиксельно
            IplImage *result_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            double start = get_time_ms();
            applyFilterParallelPixelwise(img, result_pixel, &padded_filters[j]);
            double end = get_time_ms();
            times[j][0][valid_count] = end - start;

            // По строкам
            IplImage *result_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByRows(img, result_rows, &padded_filters[j]);
            end = get_time_ms();
            times[j][1][valid_count] = end - start;

            // По столбцам
            IplImage *result_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByCols(img, result_cols, &padded_filters[j]);
            end = get_time_ms();
            times[j][2][valid_count] = end - start;

            // Блоки 32x32
            IplImage *result_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByBlocks32(img, result_32, &padded_filters[j]);
            end = get_time_ms();
            times[j][3][valid_count] = end - start;

            // Блоки 64x64
            IplImage *result_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByBlocks64(img, result_64, &padded_filters[j]);
            end = get_time_ms();
            times[j][4][valid_count] = end - start;

            // Блоки 128x128
            IplImage *result_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByBlocks128(img, result_128, &padded_filters[j]);
            end = get_time_ms();
            times[j][5][valid_count] = end - start;

            // Проверяем, что все стратегии дают одинаковый результат
            int eq = imagesEqual(result_pixel, result_rows);
            if (eq)
                eq = imagesEqual(result_pixel, result_cols);
            if (eq)
                eq = imagesEqual(result_pixel, result_32);
            if (eq)
                eq = imagesEqual(result_pixel, result_64);
            if (eq)
                eq = imagesEqual(result_pixel, result_128);

            if (!eq)
            {
                printf("ERROR: Filter %s on image %d: results do not match!\n", filter_names[j], i);
            }
            assert(eq);

            cvReleaseImage(&result_pixel);
            cvReleaseImage(&result_rows);
            cvReleaseImage(&result_cols);
            cvReleaseImage(&result_32);
            cvReleaseImage(&result_64);
            cvReleaseImage(&result_128);
        }

        cvReleaseImage(&img);
        valid_count++;
    }

    for (int j = 0; j < 5; j++)
    {
        filter_free(&padded_filters[j]);
        filter_free(&original_filters[j]);
    }

    printf("\n                                        ZERO PADDING RESULTS (PARALLEL ONLY)\n\n");
    for (int j = 0; j < 5; j++)
    {
        printf("%s:\n", filter_names[j]);
        for (int i = 0; i < valid_count; i++)
        {
            printf("│ Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for \t %-30s \n",
                   times[j][0][i], times[j][1][i], times[j][2][i], times[j][3][i], times[j][4][i], times[j][5][i], imageNames[i]);
        }
        printf("\n");
    }

    printf("\n                                        TEST 3 PASSED\n");
}

void testZeroFilter(void)
{
    printf("\n");
    printf("                                        TEST 4: ZERO FILTER                                         \n");
    printf("\n");

    double times[6][15] = {0}; // 6 стратегий (без Seq)
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

        // Попиксельно
        IplImage *result_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilterParallelPixelwise(img, result_pixel, &zero_filter);
        double end = get_time_ms();
        times[0][valid_count] = end - start;

        // По строкам
        IplImage *result_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, result_rows, &zero_filter);
        end = get_time_ms();
        times[1][valid_count] = end - start;

        // По столбцам
        IplImage *result_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, result_cols, &zero_filter);
        end = get_time_ms();
        times[2][valid_count] = end - start;

        // Блоки 32x32
        IplImage *result_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks32(img, result_32, &zero_filter);
        end = get_time_ms();
        times[3][valid_count] = end - start;

        // Блоки 64x64
        IplImage *result_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks64(img, result_64, &zero_filter);
        end = get_time_ms();
        times[4][valid_count] = end - start;

        // Блоки 128x128
        IplImage *result_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByBlocks128(img, result_128, &zero_filter);
        end = get_time_ms();
        times[5][valid_count] = end - start;

        // Проверка: все пиксели должны быть чёрными (0,0,0)
        int step = result_pixel->widthStep;
        int channels = result_pixel->nChannels;
        const unsigned char *data = (const unsigned char *)result_pixel->imageData;
        int all_black = 1;
        for (int y = 0; y < result_pixel->height && all_black; y++)
        {
            for (int x = 0; x < result_pixel->width; x++)
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

        cvReleaseImage(&result_pixel);
        cvReleaseImage(&result_rows);
        cvReleaseImage(&result_cols);
        cvReleaseImage(&result_32);
        cvReleaseImage(&result_64);
        cvReleaseImage(&result_128);
        cvReleaseImage(&img);

        valid_count++;
    }

    filter_free(&zero_filter);

    printf("\n                                        ZERO FILTER RESULTS (PARALLEL ONLY)\n\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for \t %-30s \n",
               times[0][i], times[1][i], times[2][i], times[3][i], times[4][i], times[5][i], imageNames[i]);
    }

    double total_pixel = 0, total_rows = 0, total_cols = 0;
    double total_32 = 0, total_64 = 0, total_128 = 0;
    for (int i = 0; i < valid_count; i++)
    {
        total_pixel += times[0][i];
        total_rows += times[1][i];
        total_cols += times[2][i];
        total_32 += times[3][i];
        total_64 += times[4][i];
        total_128 += times[5][i];
    }
    printf("\n");
    printf("Total: \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms \n",
           total_pixel, total_rows, total_cols, total_32, total_64, total_128);

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