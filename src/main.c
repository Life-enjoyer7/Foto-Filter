#include <time.h>
#include <getopt.h>
#include "filter.h"

#ifndef CV_LOAD_IMAGE_COLOR
#define CV_LOAD_IMAGE_COLOR 1
#endif

#define NUM_FILTERS 15

double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
};

int main(int argc, char *argv[])
{
    int filterId = -1;
    int strategyId = -1;
    char *load_path = NULL;
    char *save_path = NULL;

    Filter filters[NUM_FILTERS];
    filters[0] = filter_blur3x3();
    filters[1] = filter_blur5x5();
    filters[2] = filter_gaussian3x3();
    filters[3] = filter_gaussian5x5();
    filters[4] = filter_motionblur();
    filters[5] = filter_findedges1();
    filters[6] = filter_findedges2();
    filters[7] = filter_findedges3();
    filters[8] = filter_findedges4();
    filters[9] = filter_sharpen1();
    filters[10] = filter_sharpen2();
    filters[11] = filter_sharpen3();
    filters[12] = filter_emboss1();
    filters[13] = filter_emboss2();
    filters[14] = filter_identity();

    void (*strategies[7])(const IplImage *, IplImage *, const Filter *) = {
        applyFilter,
        applyFilterParallelPixelwise,
        applyFilterParallelByRows,
        applyFilterParallelByCols,
        applyFilterParallelByBlocks32,
        applyFilterParallelByBlocks64,
        applyFilterParallelByBlocks128};

    static struct option long_options[] = {
        {"filter", required_argument, 0, 'f'},
        {"tactic", required_argument, 0, 't'},
        {"src", required_argument, 0, 's'},
        {"out", required_argument, 0, 'o'},
        {0, 0, 0, 0}};

    int c;
    int option_index = 0;

    while ((c = getopt_long(argc, argv, "f:t:s:o:", long_options, &option_index)) != -1)
    {
        switch (c)
        {
        case 'f':
            filterId = atoi(optarg);
            break;
        case 't':
            strategyId = atoi(optarg);
            break;
        case 's':
            load_path = optarg;
            break;
        case 'o':
            save_path = optarg;
            break;
        default:
            printf("Unknown option: %c\n", c);
            return 1;
        }
    }

    if (filterId < 0 || filterId >= NUM_FILTERS)
    {
        printf("Error: Invalid filter ID (0-14)\n");
        return 1;
    }

    if (strategyId < 0 || strategyId >= 8)
    {
        printf("Error: Invalid strategy ID (0-3)\n");
        return 1;
    }

    if (!load_path)
    {
        printf("Error: Source image required (-s)\n");
        return 1;
    }

    if (!save_path)
    {
        printf("Error: Output path required (-o)\n");
        return 1;
    }

    printf("Loading image from: '%s'\n", load_path);
    IplImage *image = cvLoadImage(load_path, CV_LOAD_IMAGE_COLOR);
    if (!image)
    {
        printf("Error: Failed to load image\n");
        return 1;
    }

    IplImage *result = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 3);

    printf("Applying filter %d with strategy %d...\n", filterId, strategyId);

    double start = get_time_ms();

    strategies[strategyId](image, result, &filters[filterId]);

    double end = get_time_ms();

    printf("Saving to: '%s'\n", save_path);
    cvSaveImage(save_path, result);
    printf("Completed with the time spent applying the filter equal to %8.2f ms\n", end - start);

    cvReleaseImage(&image);
    cvReleaseImage(&result);
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        filter_free(&filters[i]);
    }

    return 0;
}