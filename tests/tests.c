#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include "../src/filter.h"
#include "../src/pipeline.h"
#include "../src/utils.h"

void testIdentityFilter(void)
{
    printf("\n");
    printf("                        TEST 1: IDENTITY FILTER                                 \n");
    printf("                   (Pipeline vs Sequential Comparison)                         \n");

    Filter identity = filter_identity();

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }

    printf("Found %d images in %s\n", num_images, input_dir);

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // ЗАМЕР КОНВЕЙЕРА
    printf("\nRunning pipeline on all images...\n");
    double pipeline_start = get_time_ms();

    pipeline_run(input_paths, output_paths, num_images, 14, 1, 10);

    double pipeline_end = get_time_ms();
    double pipeline_time = pipeline_end - pipeline_start;
    printf("Pipeline completed in %.2f ms\n", pipeline_time);

    //  ПОСЛЕДОВАТЕЛЬНАЯ ОБРАБОТКА
    int valid_count = 0;
    double seq_total_time = 0;

    printf("\nComparing pipeline results with sequential processing...\n");

    for (int j = 0; j < num_images; j++)
    {
        IplImage *img = cvLoadImage(input_paths[j], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %s\n", input_paths[j]);
            continue;
        }

        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double seq_start = get_time_ms();
        applyFilter(img, result_seq, &identity);
        double seq_time = get_time_ms() - seq_start;
        seq_total_time += seq_time;

        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
        if (!pipeline_img)
        {
            printf("ERROR: Failed to load pipeline result: %s\n", output_paths[j]);
            cvReleaseImage(&result_seq);
            cvReleaseImage(&img);
            continue;
        }

        int eq = imagesEqual(pipeline_img, result_seq);
        if (!eq)
        {
            printf("ERROR: DIFFERENT images");
        }
        assert(eq);

        cvReleaseImage(&result_seq);
        cvReleaseImage(&pipeline_img);
        cvReleaseImage(&img);
        valid_count++;
    }

    printf("\n");

    printf("                          PIPELINE vs SEQUENTIAL RESULTS                       \n");

    printf("  Pipeline (10 workers): %.2f ms\n", pipeline_time);
    printf("  Sequential (total):   %.2f ms\n", seq_total_time);
    printf("  Speedup:              %.2fx\n", seq_total_time / pipeline_time);
    printf("  Processed images:     %d\n", valid_count);

    printf("                          TEST 1 PASSED                                        \n");

    // Очистка
    filter_free(&identity);
    for (int i = 0; i < num_images; i++)
    {
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(input_paths);
    free(output_paths);
}

void testShiftComposition(void)
{
    printf("\n");

    printf("                        TEST 2: SHIFT COMPOSITION                              \n");
    printf("                   (Pipeline vs Sequential Comparison)                         \n");

    // Создаём фильтры сдвига (используя filter.h)
    Filter shiftRight = filter_shift_right();
    Filter shiftLeft = filter_shift_left();
    Filter shiftUp = filter_shift_up();
    Filter shiftDown = filter_shift_down();
    Filter shiftDiagUp = filter_shift_diag_up();
    Filter shiftDiagDown = filter_shift_diag_down();

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }

    printf("Found %d images in %s\n", num_images, input_dir);

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // Массивы для хранения результатов
    double pipeline_times[3] = {0}; // 0: Right-Left, 1: Up-Down, 2: Diag
    double sequential_times[3] = {0};

    // Массивы для проверки корректности

    //  RIGHT-LEFT
    printf("\n");
    printf("\nProcessing RIGHT-LEFT COMPOSITION\n");
    printf("\n");
    // Pipeline
    double pipe_start = get_time_ms();
    pipeline_run(input_paths, output_paths, num_images, 15, 1, 10);
    pipeline_run(output_paths, output_paths, num_images, 16, 1, 10);
    pipeline_times[0] = get_time_ms() - pipe_start;
    printf("\n");
    printf("Pipeline (Right-Left) completed\n");
    printf("\n");
    // Sequential
    double seq_total = 0;

    for (int j = 0; j < num_images; j++)
    {
        IplImage *img = cvLoadImage(input_paths[j], 1);
        if (!img)
            continue;

        IplImage *temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

        double seq_start = get_time_ms();
        applyFilter(img, temp, &shiftRight);
        applyFilter(temp, result_seq, &shiftLeft);
        seq_total += get_time_ms() - seq_start;

        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);

        int eq = imagesEqual(pipeline_img, result_seq);
        if (!eq)
        {
            printf("ERROR: DIFFERENT images");
        }
        assert(eq);

        cvReleaseImage(&temp);
        cvReleaseImage(&result_seq);
        cvReleaseImage(&pipeline_img);
        cvReleaseImage(&img);
    }
    sequential_times[0] = seq_total;

    //  UP-DOWN
    printf("\n");
    printf("\nProcessing UP-DOWN COMPOSITION\n");
    printf("\n");
    pipe_start = get_time_ms();
    pipeline_run(input_paths, output_paths, num_images, 17, 1, 10);
    pipeline_run(output_paths, output_paths, num_images, 18, 1, 10);
    pipeline_times[1] = get_time_ms() - pipe_start;
    printf("\n");
    printf("Pipeline (Up-Down) completed\n");
    printf("\n");
    seq_total = 0;

    for (int j = 0; j < num_images; j++)
    {
        IplImage *img = cvLoadImage(input_paths[j], 1);
        if (!img)
            continue;

        IplImage *temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

        double seq_start = get_time_ms();
        applyFilter(img, temp, &shiftUp);
        applyFilter(temp, result_seq, &shiftDown);
        seq_total += get_time_ms() - seq_start;

        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);

        int eq = imagesEqual(pipeline_img, result_seq);
        if (!eq)
        {
            printf("ERROR: DIFFERENT images");
        }
        assert(eq);

        cvReleaseImage(&temp);
        cvReleaseImage(&result_seq);
        cvReleaseImage(&pipeline_img);
        cvReleaseImage(&img);
    }
    sequential_times[1] = seq_total;

    //  DIAG
    printf("\n");
    printf("\nProcessing DIAG COMPOSITION \n");
    printf("\n");
    pipe_start = get_time_ms();
    pipeline_run(input_paths, output_paths, num_images, 19, 1, 10);
    pipeline_run(output_paths, output_paths, num_images, 20, 1, 10);
    pipeline_times[2] = get_time_ms() - pipe_start;
    printf("\n");
    printf("Pipeline (Diag) completed\n");
    printf("\n");
    seq_total = 0;

    for (int j = 0; j < num_images; j++)
    {
        IplImage *img = cvLoadImage(input_paths[j], 1);
        if (!img)
            continue;

        IplImage *temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

        double seq_start = get_time_ms();
        applyFilter(img, temp, &shiftDiagUp);
        applyFilter(temp, result_seq, &shiftDiagDown);
        seq_total += get_time_ms() - seq_start;

        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);

        int eq = imagesEqual(pipeline_img, result_seq);
        if (!eq)
        {
            printf("ERROR: DIFFERENT images");
        }
        assert(eq);

        cvReleaseImage(&temp);
        cvReleaseImage(&result_seq);
        cvReleaseImage(&pipeline_img);
        cvReleaseImage(&img);
    }
    sequential_times[2] = seq_total;

    printf("\n");
    printf("                          SHIFT COMPOSITION RESULTS                           \n");
    printf("\n");
    printf("  Right-Left Pipeline:   %.2f ms\n", pipeline_times[0]);
    printf("  Right-Left Sequential: %.2f ms\n", sequential_times[0]);
    printf("  Right-Left Speedup:    %.2fx\n", sequential_times[0] / pipeline_times[0]);

    printf("\n");
    printf("  Up-Down Pipeline:      %.2f ms\n", pipeline_times[1]);
    printf("  Up-Down Sequential:    %.2f ms\n", sequential_times[1]);
    printf("  Up-Down Speedup:       %.2fx\n", sequential_times[1] / pipeline_times[1]);

    printf("\n");
    printf("  Diag Pipeline:         %.2f ms\n", pipeline_times[2]);
    printf("  Diag Sequential:       %.2f ms\n", sequential_times[2]);
    printf("  Diag Speedup:          %.2fx\n", sequential_times[2] / pipeline_times[2]);

    printf("\n");
    printf("  Processed images:      %d\n", num_images);
    printf("\n");
    printf("                          TEST 2 PASSED                                        \n");

    // Очистка
    filter_free(&shiftRight);
    filter_free(&shiftLeft);
    filter_free(&shiftUp);
    filter_free(&shiftDown);
    filter_free(&shiftDiagUp);
    filter_free(&shiftDiagDown);

    for (int i = 0; i < num_images; i++)
    {
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(input_paths);
    free(output_paths);
}

void testZeroPadding(void)
{
    printf("\n");
    printf("                        TEST 3: ZERO PADDING                                    \n");
    printf("                   (Pipeline vs Sequential Comparison)                         \n");

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }
    printf("\n");
    printf("Found %d images in %s\n", num_images, input_dir);
    printf("\n");
    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // Индексы padded фильтров в pipeline (нужно определить в pipeline.c)
    int padded_indices[5] = {21, 22, 23, 24, 25}; // blur, gaussian, findedges1, sharpen1, emboss1
    const char *filter_names[5] = {"blur3x3", "gaussian3x3", "findedges1", "sharpen1", "emboss1"};

    // Оригинальные фильтры (для последовательной обработки)
    Filter original_filters[5];
    original_filters[0] = filter_blur3x3();
    original_filters[1] = filter_gaussian3x3();
    original_filters[2] = filter_findedges1();
    original_filters[3] = filter_sharpen1();
    original_filters[4] = filter_emboss1();

    // Массивы для хранения результатов по каждому фильтру
    double pipeline_times[5] = {0};
    double sequential_times[5] = {0};
    int valid_counts[5] = {0};

    // ОБРАБОТКА КАЖДОГО ФИЛЬТРА
    for (int f = 0; f < 5; f++)
    {
        printf("\n");
        printf("\nProcessing filter: %s\n", filter_names[f]);
        printf("\n");
        // Pipeline (используем padded фильтр из pipeline по индексу)
        double pipe_start = get_time_ms();
        pipeline_run(input_paths, output_paths, num_images, padded_indices[f], 1, 10);
        pipeline_times[f] = get_time_ms() - pipe_start;
        printf("Pipeline (%s) completed\n", filter_names[f]);

        // Sequential (оригинальный фильтр)
        double seq_total = 0;
        int valid = 0;
        for (int j = 0; j < num_images; j++)
        {
            IplImage *img = cvLoadImage(input_paths[j], 1);
            if (!img)
                continue;

            IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            double seq_start = get_time_ms();
            applyFilter(img, result_seq, &original_filters[f]);
            seq_total += get_time_ms() - seq_start;

            IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);

            // Проверяем, что результаты совпадают
            int eq = imagesEqual(pipeline_img, result_seq);
            if (!eq)
            {
                printf("ERROR: DIFFERENT images");
            }
            assert(eq);

            cvReleaseImage(&result_seq);
            cvReleaseImage(&pipeline_img);
            cvReleaseImage(&img);
        }
        sequential_times[f] = seq_total;
        valid_counts[f] = valid;
    }

    // ВЫВОД РЕЗУЛЬТАТОВ
    printf("\n");
    printf("                          ZERO PADDING RESULTS                                 \n");
    printf("\n");

    for (int f = 0; f < 5; f++)
    {
        double speedup = sequential_times[f] / pipeline_times[f];
        const char *correct_str = (valid_counts[f] == num_images) ? "YES" : "NO";
        printf("  %-12s Pipeline:   %.2f ms\n", filter_names[f], pipeline_times[f]);
        printf("  %-12s Sequential: %.2f ms\n", filter_names[f], sequential_times[f]);
        printf("  %-12s Speedup:    %.2fx\n", filter_names[f], speedup);
        printf("  %-12s Correct:    %s\n", filter_names[f], correct_str);
        printf("\n");
    }

    printf("  Processed images:      %d\n", num_images);
    printf("\n");
    printf("                          TEST 3 PASSED                                        \n");

    // Очистка
    for (int i = 0; i < 5; i++)
    {
        filter_free(&original_filters[i]);
    }
    for (int i = 0; i < num_images; i++)
    {
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(input_paths);
    free(output_paths);
}

void testZeroFilter(void)
{
    printf("\n");
    printf("                        TEST 4: ZERO FILTER                                     \n");
    printf("                   (Pipeline vs Sequential Comparison)                         \n");

    Filter zero = filter_zero();

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }
    printf("\n");
    printf("Found %d images in %s\n", num_images, input_dir);
    printf("\n");
    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // ЗАМЕР КОНВЕЙЕРА
    printf("\nRunning pipeline on all images...\n");
    double pipeline_start = get_time_ms();

    pipeline_run(input_paths, output_paths, num_images, 26, 1, 10); // 26 — индекс zero_filter в pipeline

    double pipeline_end = get_time_ms();
    double pipeline_time = pipeline_end - pipeline_start;
    printf("Pipeline completed in %.2f ms\n", pipeline_time);

    // ПОСЛЕДОВАТЕЛЬНАЯ ОБРАБОТКА
    int valid_count = 0;
    double seq_total_time = 0;
    printf("\n");
    printf("\nComparing pipeline results with sequential processing...\n");
    printf("\n");
    for (int j = 0; j < num_images; j++)
    {
        IplImage *img = cvLoadImage(output_paths[j], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %s\n", output_paths[j]);
            continue;
        }

        // Проверяем, что изображение полностью чёрное (все каналы = 0)
        int step = img->widthStep;
        int channels = img->nChannels;
        const unsigned char *data = (const unsigned char *)img->imageData;
        int all_black = 1;
        for (int y = 0; y < img->height && all_black; y++)
        {
            for (int x = 0; x < img->width; x++)
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
            printf("ERROR: Zero filter on image %d: result is not all black!\n", j);
        }
        assert(all_black);

        // Последовательная обработка (ноль фильтр)
        IplImage *result_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double seq_start = get_time_ms();
        applyFilter(img, result_seq, &zero);
        double seq_time = get_time_ms() - seq_start;
        seq_total_time += seq_time;

        // Результат конвейера
        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
        if (!pipeline_img)
        {
            printf("ERROR: Failed to load pipeline result: %s\n", output_paths[j]);
            cvReleaseImage(&result_seq);
            cvReleaseImage(&img);
            continue;
        }

        int eq = imagesEqual(pipeline_img, result_seq);
        if (!eq)
        {
            printf("ERROR: DIFFERENT images");
        }
        assert(eq);

        cvReleaseImage(&result_seq);
        cvReleaseImage(&pipeline_img);
        cvReleaseImage(&img);
        valid_count++;
    }

    printf("\n");
    printf("                          PIPELINE vs SEQUENTIAL RESULTS                       \n");
    printf("\n");
    printf("  Pipeline (10 workers): %.2f ms\n", pipeline_time);
    printf("  Sequential (total):   %.2f ms\n", seq_total_time);
    printf("  Speedup:              %.2fx\n", seq_total_time / pipeline_time);
    printf("  Processed images:     %d\n", valid_count);
    printf("\n");
    printf("                          TEST 4 PASSED                                        \n");

    // Очистка
    filter_free(&zero);
    for (int i = 0; i < num_images; i++)
    {
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(input_paths);
    free(output_paths);
}

int main(void)
{
    testIdentityFilter();
    testShiftComposition();
    testZeroPadding();
    testZeroFilter();
    return 0;
}