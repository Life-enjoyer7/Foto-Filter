#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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

// Измерение времени в миллисекундах
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

    // Данные для всех трёх стратегий
    double times_seq[15] = {0};
    double times_pixel[15] = {0};
    double times_rows[15] = {0};
    double times_cols[15] = {0};

    int valid_count = 0;
    int sizes[15][2] = {0}; // хранение размеров для вывода

    for (int j = 0; j < 15; j++)
    {

        IplImage *img = cvLoadImage(imagePaths[j], 1);

        if (!img)
        {
            printf("ERROR: Failed to load image\n");
            continue;
        }

        sizes[valid_count][0] = img->width;
        sizes[valid_count][1] = img->height;

        // Последовательная версия
        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilter(img, result_seq, &identity);
        double end = get_time_ms();
        times_seq[valid_count] = end - start;

        // Параллельная версия (попиксельно)
        IplImage *result_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, result_pixel, &identity);
        end = get_time_ms();
        times_pixel[valid_count] = end - start;

        // Параллельная версия (по строкам)
        IplImage *result_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, result_rows, &identity);
        end = get_time_ms();
        times_rows[valid_count] = end - start;

        // Параллельная версия (по столбцам)
        IplImage *result_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, result_cols, &identity);
        end = get_time_ms();
        times_cols[valid_count] = end - start;

        // Проверка результатов
        int eq_seq = imagesEqual(img, result_seq);
        int eq_pixel = imagesEqual(img, result_pixel);
        int eq_rows = imagesEqual(img, result_rows);
        int eq_cols = imagesEqual(img, result_cols);

        if (!eq_seq || !eq_pixel || !eq_rows || !eq_cols)
        {
            printf("Results do not match original!\n");
        }
        assert(eq_seq && eq_pixel && eq_rows && eq_cols);

        cvReleaseImage(&result_seq);
        cvReleaseImage(&result_pixel);
        cvReleaseImage(&result_rows);
        cvReleaseImage(&result_cols);
        cvReleaseImage(&img);

        valid_count++;
    }

    // Вывод результатов в виде таблицы
    printf("\n");

    printf("                                        PERFORMANCE RESULTS                                         \n");

    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for  \t %-30s \n",
               times_seq[i], times_pixel[i], times_rows[i], times_cols[i], imageNames[i]);
    }

    // Итоговые суммы и ускорения
    double total_seq = 0, total_pixel = 0, total_rows = 0, total_cols = 0;
    for (int i = 0; i < valid_count; i++)
    {
        total_seq += times_seq[i];
        total_pixel += times_pixel[i];
        total_rows += times_rows[i];
        total_cols += times_cols[i];
    }
    printf("\n");
    printf("Total: \t Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \n",
           total_seq, total_pixel, total_rows, total_cols);

    printf("\n");

    printf("                                        TEST 1 PASSED                                  \n");
}

void testShiftComposition(void)
{
    printf("\n");
    printf("                                        TEST 2: SHIFT COMPOSITION                         \n");

    // Структура для хранения результатов: [композиция][стратегия][изображение]
    // композиция: 0=RightLeft, 1=UpDown, 2=Diag
    // стратегия: 0=Seq, 1=Pixel, 2=Rows, 3=Cols
    double times[3][4][15] = {0};
    int valid_count = 0;
    int sizes[15][2] = {0};

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        sizes[valid_count][0] = img->width;
        sizes[valid_count][1] = img->height;

        // Фильтры сдвига
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

        // ==================== Right-Left ====================
        // Последовательная
        IplImage *temp_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilter(img, temp_seq, &shiftRight);
        applyFilter(temp_seq, final_seq, &shiftLeft);
        double end = get_time_ms();
        times[0][0][valid_count] = end - start;
        int eq_seq = imagesEqual(img, final_seq);

        // Параллельная попиксельно
        IplImage *temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, temp_pixel, &shiftRight);
        applyFilterParallelPixelwise(temp_pixel, final_pixel, &shiftLeft);
        end = get_time_ms();
        times[0][1][valid_count] = end - start;
        int eq_pixel = imagesEqual(img, final_pixel);

        // Параллельная по строкам
        IplImage *temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, temp_rows, &shiftRight);
        applyFilterParallelByRows(temp_rows, final_rows, &shiftLeft);
        end = get_time_ms();
        times[0][2][valid_count] = end - start;
        int eq_rows = imagesEqual(img, final_rows);

        // Параллельная по столбцам
        IplImage *temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, temp_cols, &shiftRight);
        applyFilterParallelByCols(temp_cols, final_cols, &shiftLeft);
        end = get_time_ms();
        times[0][3][valid_count] = end - start;
        int eq_cols = imagesEqual(img, final_cols);

        if (!eq_seq || !eq_pixel || !eq_rows || !eq_cols)
        {
            printf("Right-Left composition failed for image %d\n", i);
        }
        assert(eq_seq && eq_pixel && eq_rows && eq_cols);

        cvReleaseImage(&temp_seq);
        cvReleaseImage(&final_seq);
        cvReleaseImage(&temp_pixel);
        cvReleaseImage(&final_pixel);
        cvReleaseImage(&temp_rows);
        cvReleaseImage(&final_rows);
        cvReleaseImage(&temp_cols);
        cvReleaseImage(&final_cols);

        // ==================== Up-Down ====================
        temp_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilter(img, temp_seq, &shiftUp);
        applyFilter(temp_seq, final_seq, &shiftDown);
        end = get_time_ms();
        times[1][0][valid_count] = end - start;
        eq_seq = imagesEqual(img, final_seq);

        temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, temp_pixel, &shiftUp);
        applyFilterParallelPixelwise(temp_pixel, final_pixel, &shiftDown);
        end = get_time_ms();
        times[1][1][valid_count] = end - start;
        eq_pixel = imagesEqual(img, final_pixel);

        temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, temp_rows, &shiftUp);
        applyFilterParallelByRows(temp_rows, final_rows, &shiftDown);
        end = get_time_ms();
        times[1][2][valid_count] = end - start;
        eq_rows = imagesEqual(img, final_rows);

        temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, temp_cols, &shiftUp);
        applyFilterParallelByCols(temp_cols, final_cols, &shiftDown);
        end = get_time_ms();
        times[1][3][valid_count] = end - start;
        eq_cols = imagesEqual(img, final_cols);

        if (!eq_seq || !eq_pixel || !eq_rows || !eq_cols)
        {
            printf("Up-Down composition failed for image %d\n", i);
        }
        assert(eq_seq && eq_pixel && eq_rows && eq_cols);

        cvReleaseImage(&temp_seq);
        cvReleaseImage(&final_seq);
        cvReleaseImage(&temp_pixel);
        cvReleaseImage(&final_pixel);
        cvReleaseImage(&temp_rows);
        cvReleaseImage(&final_rows);
        cvReleaseImage(&temp_cols);
        cvReleaseImage(&final_cols);

        // ==================== Diag ====================
        temp_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilter(img, temp_seq, &shiftDiagUp);
        applyFilter(temp_seq, final_seq, &shiftDiagDown);
        end = get_time_ms();
        times[2][0][valid_count] = end - start;
        eq_seq = imagesEqual(img, final_seq);

        temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, temp_pixel, &shiftDiagUp);
        applyFilterParallelPixelwise(temp_pixel, final_pixel, &shiftDiagDown);
        end = get_time_ms();
        times[2][1][valid_count] = end - start;
        eq_pixel = imagesEqual(img, final_pixel);

        temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, temp_rows, &shiftDiagUp);
        applyFilterParallelByRows(temp_rows, final_rows, &shiftDiagDown);
        end = get_time_ms();
        times[2][2][valid_count] = end - start;
        eq_rows = imagesEqual(img, final_rows);

        temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, temp_cols, &shiftDiagUp);
        applyFilterParallelByCols(temp_cols, final_cols, &shiftDiagDown);
        end = get_time_ms();
        times[2][3][valid_count] = end - start;
        eq_cols = imagesEqual(img, final_cols);

        if (!eq_seq || !eq_pixel || !eq_rows || !eq_cols)
        {
            printf("Diag composition failed for image %d\n", i);
        }
        assert(eq_seq && eq_pixel && eq_rows && eq_cols);

        cvReleaseImage(&temp_seq);
        cvReleaseImage(&final_seq);
        cvReleaseImage(&temp_pixel);
        cvReleaseImage(&final_pixel);
        cvReleaseImage(&temp_rows);
        cvReleaseImage(&final_rows);
        cvReleaseImage(&temp_cols);
        cvReleaseImage(&final_cols);

        // Освобождение фильтров и изображения
        filter_free(&shiftRight);
        filter_free(&shiftLeft);
        filter_free(&shiftUp);
        filter_free(&shiftDown);
        filter_free(&shiftDiagUp);
        filter_free(&shiftDiagDown);
        cvReleaseImage(&img);

        valid_count++;
    }

    // Вывод результатов в виде таблицы
    printf("\n");
    printf("                                        SHIFT COMPOSITION RESULTS                                          \n");
    printf("\n");
    printf("LEFT-RIGHT                                         \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[0][0][i], times[0][1][i], times[0][2][i], times[0][3][i], imageNames[i]);
    };
    printf("\n");
    printf("UD-DOWN                                        \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[1][0][i], times[1][1][i], times[1][2][i], times[1][3][i], imageNames[i]);
    };
    printf("\n");
    printf("DIAG                                       \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[2][0][i], times[2][1][i], times[2][2][i], times[2][3][i], imageNames[i]);
    };
    double total_seq = 0, total_pixel = 0, total_rows = 0, total_cols = 0;
    for (int i = 0; i < 15; i++)
    {
        total_seq += times[0][0][i];
        total_pixel += times[0][1][i];
        total_rows += times[0][2][i];
        total_cols += times[0][3][i];
    }
    printf("\n");
    printf("Total: \t Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \n",
           total_seq, total_pixel, total_rows, total_cols);

    printf("\n");
    printf("                                        TEST 2 PASSED                                  \n");
}

void testZeroPadding(void)
{
    printf("\n");
    printf("                                        TEST 3: ZERO PADDING                                         \n");
    printf("\n");

    // 5 фильтров: 0=blur3x3, 1=gaussian3x3, 2=findedges1, 3=sharpen1, 4=emboss1
    // 4 стратегии: 0=Seq, 1=Pixel, 2=Rows, 3=Cols
    double times[5][4][15] = {0};
    int valid_count = 0;
    int sizes[15][2] = {0};

    // Оригинальные фильтры
    Filter original_filters[5];
    original_filters[0] = filter_blur3x3();
    original_filters[1] = filter_gaussian3x3();
    original_filters[2] = filter_findedges1();
    original_filters[3] = filter_sharpen1();
    original_filters[4] = filter_emboss1();

    // Padded фильтры (расширенные нулями)
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

    const char *filter_names[5] = {
        "blur3x3",
        "gaussian3x3",
        "findedges1",
        "sharpen1",
        "emboss1"};

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        sizes[valid_count][0] = img->width;
        sizes[valid_count][1] = img->height;

        for (int j = 0; j < 5; j++)
        {
            // Последовательная версия
            IplImage *result_orig = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            double start = get_time_ms();
            applyFilter(img, result_orig, &original_filters[j]);
            double end = get_time_ms();
            times[j][0][valid_count] = end - start;

            // Параллельная попиксельно
            IplImage *result_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelPixelwise(img, result_pixel, &padded_filters[j]);
            end = get_time_ms();
            times[j][1][valid_count] = end - start;

            // Параллельная по строкам
            IplImage *result_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByRows(img, result_rows, &padded_filters[j]);
            end = get_time_ms();
            times[j][2][valid_count] = end - start;

            // Параллельная по столбцам
            IplImage *result_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            start = get_time_ms();
            applyFilterParallelByCols(img, result_cols, &padded_filters[j]);
            end = get_time_ms();
            times[j][3][valid_count] = end - start;

            // Проверка: оригинальный фильтр должен совпадать с padded
            int eq_orig = imagesEqual(result_orig, result_pixel);
            int eq_pixel = imagesEqual(result_orig, result_rows);
            int eq_rows = imagesEqual(result_orig, result_cols);

            if (!eq_orig || !eq_pixel || !eq_rows)
            {
                printf("  ERROR: Filter %s on image %d: results do not match!\n", filter_names[j], i);
            }
            assert(eq_orig && eq_pixel && eq_rows);

            cvReleaseImage(&result_orig);
            cvReleaseImage(&result_pixel);
            cvReleaseImage(&result_rows);
            cvReleaseImage(&result_cols);
        }

        cvReleaseImage(&img);
        valid_count++;
    }
    for (int j = 0; j < 5; j++)
    {
        filter_free(&padded_filters[j]);
        filter_free(&original_filters[j]);
    }

    printf("\n");
    printf("                                        ZERO PADDING RESULTS                                          \n");
    printf("\n");
    printf("BLUR 3x3                                         \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[0][0][i], times[0][1][i], times[0][2][i], times[0][3][i], imageNames[i]);
    };
    printf("\n");
    printf("GAUSSIAN 3x3                                        \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[1][0][i], times[1][1][i], times[1][2][i], times[1][3][i], imageNames[i]);
    };
    printf("\n");
    printf("FIND EDGES 1                                       \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[2][0][i], times[2][1][i], times[2][2][i], times[2][3][i], imageNames[i]);
    };
    printf("\n");
    printf("SHARPEN 1                                       \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[3][0][i], times[3][1][i], times[3][2][i], times[3][3][i], imageNames[i]);
    };
    printf("\n");
    printf("EMBOSS 1                                       \n");
    printf("\n");
    for (int i = 0; i < 15; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for \t %-30s \n",
               times[4][0][i], times[4][1][i], times[4][2][i], times[4][3][i], imageNames[i]);
    };
    double total_seq = 0, total_pixel = 0, total_rows = 0, total_cols = 0;
    for (int i = 0; i < 15; i++)
    {
        total_seq += times[0][0][i] + times[1][0][i] + times[2][0][i] + times[3][0][i] + times[4][0][i];
        total_pixel += times[0][1][i] + times[1][1][i] + times[2][1][i] + times[3][1][i] + times[4][1][i];
        total_rows += times[0][2][i] + times[1][2][i] + times[2][2][i] + times[3][2][i] + times[3][2][i];
        total_cols += times[0][3][i] + times[1][3][i] + times[2][3][i] + times[3][3][i] + times[4][3][i];
    }
    printf("\n");
    printf("Total: \t Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \n",
           total_seq, total_pixel, total_rows, total_cols);

    printf("\n");
    printf("                                        TEST 3 PASSED                                  \n");
}

void testZeroFilter(void)
{
    printf("\n");
    printf("                                        TEST 4: ZERO FILTER                                         \n");
    printf("\n");

    // 4 стратегии: 0=Seq, 1=Pixel, 2=Rows, 3=Cols
    double times[4][15] = {0};
    int valid_count = 0;
    int sizes[15][2] = {0};

    // Нулевой фильтр (все нули)
    double kernel_zero[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}};

    Filter zero_filter = filter_create(3, 3, &kernel_zero[0][0], 1.0, 0.0);

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        sizes[valid_count][0] = img->width;
        sizes[valid_count][1] = img->height;

        // Последовательная версия
        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double start = get_time_ms();
        applyFilter(img, result_seq, &zero_filter);
        double end = get_time_ms();
        times[0][valid_count] = end - start;

        // Проверка: все пиксели должны быть чёрными (0,0,0)
        int step = result_seq->widthStep;
        int channels = result_seq->nChannels;
        const unsigned char *data = (const unsigned char *)result_seq->imageData;
        int all_black = 1;
        for (int y = 0; y < result_seq->height && all_black; y++)
        {
            for (int x = 0; x < result_seq->width; x++)
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
            printf("  ERROR: Zero filter on image %d: result is not all black!\n", i);
        }
        assert(all_black);

        // Параллельная попиксельно
        IplImage *result_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelPixelwise(img, result_pixel, &zero_filter);
        end = get_time_ms();
        times[1][valid_count] = end - start;

        // Параллельная по строкам
        IplImage *result_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByRows(img, result_rows, &zero_filter);
        end = get_time_ms();
        times[2][valid_count] = end - start;

        // Параллельная по столбцам
        IplImage *result_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        start = get_time_ms();
        applyFilterParallelByCols(img, result_cols, &zero_filter);
        end = get_time_ms();
        times[3][valid_count] = end - start;

        // Проверка параллельных версий
        int eq_pixel = imagesEqual(result_seq, result_pixel);
        int eq_rows = imagesEqual(result_seq, result_rows);
        int eq_cols = imagesEqual(result_seq, result_cols);

        if (!eq_pixel || !eq_rows || !eq_cols)
        {
            printf("  ERROR: Zero filter on image %d: parallel results do not match sequential!\n", i);
        }
        assert(eq_pixel && eq_rows && eq_cols);

        cvReleaseImage(&result_seq);
        cvReleaseImage(&result_pixel);
        cvReleaseImage(&result_rows);
        cvReleaseImage(&result_cols);
        cvReleaseImage(&img);

        valid_count++;
    }

    filter_free(&zero_filter);

    printf("\n");
    // 4 стратегии: 0=Seq, 1=Pixel, 2=Rows, 3=Cols
    printf("                                        ZERO FILTER RESULTS                                         \n");
    printf("\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms for  \t %-30s \n",
               times[0][i], times[1][i], times[2][i], times[3][i], imageNames[i]);
    }

    // Итоговые суммы и ускорения
    double total_seq = 0, total_pixel = 0, total_rows = 0, total_cols = 0;
    for (int i = 0; i < valid_count; i++)
    {
        total_seq += times[0][i];
        total_pixel += times[1][i];
        total_rows += times[2][i];
        total_cols += times[3][i];
    }
    printf("\n");
    printf("Total: \t Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f ms \n",
           total_seq, total_pixel, total_rows, total_cols);

    printf("\n");

    printf("                                        TEST 4 PASSED                                  \n");
}

int main(void)
{
    testIdentityFilter();
    testShiftComposition();
    testZeroPadding();
    testZeroFilter();
    return 0;
}