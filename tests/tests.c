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

    printf("\nRunning pipeline on all images (4 workers)...\n");
    double pipe_start = get_time_ms();
    pipeline_run(input_paths, output_paths, num_images, 14, 1, 10);
    double pipe_time = get_time_ms() - pipe_start;
    printf("Pipeline completed in %.2f ms\n", pipe_time);

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
            printf("ERROR: Pipeline result differs from original for %s \n",
                   input_paths[j]);
        }
        assert(eq);

        cvReleaseImage(&result_seq);
        cvReleaseImage(&pipeline_img);
        cvReleaseImage(&img);
        valid_count++;
    }

    printf("\n");

    printf("                          PIPELINE vs SEQUENTIAL RESULTS                       \n");

    printf("  Pipeline (4 workers): %.2f ms\n", pipe_time);
    printf("  Sequential (total):   %.2f ms\n", seq_total_time);
    printf("  Speedup:              %.2fx\n", seq_total_time / pipe_time);
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

int main(void)
{
    testIdentityFilter();
    return 0;
}