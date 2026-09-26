#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "filter.h"
#include "main_utils.h"

#ifndef CV_LOAD_IMAGE_COLOR
#define CV_LOAD_IMAGE_COLOR 1
#endif

#define DEFAULT_REPEAT 10

int main(int argc, char *argv[])
{
    int filterId = -1;
    int strategyId = -1;
    char *load_path = NULL;
    char *save_path = NULL;
    int repeat = DEFAULT_REPEAT;

    static struct option long_options[] = {
        {"filter", required_argument, 0, 'f'},
        {"tactic", required_argument, 0, 't'},
        {"src", required_argument, 0, 's'},
        {"out", required_argument, 0, 'o'},
        {"repeat", required_argument, 0, 'r'},
        {0, 0, 0, 0}};

    int c;
    int option_index = 0;

    while ((c = getopt_long(argc, argv, "f:t:s:o:r:", long_options, &option_index)) != -1)
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
        case 'r':
            repeat = atoi(optarg);
            break;
        default:
            printf("Unknown option: %c\n", c);
            return 1;
        }
    }

    if (repeat < 1)
    {
        printf("Error: --repeat must be >= 1\n");
        return 1;
    }

    if (filterId < 0 || filterId >= NUM_FILTERS)
    {
        printf("Error: Invalid filter ID (0-14)\n");
        return 1;
    }

    if (strategyId < 0 || strategyId >= 6)
    {
        printf("Error: Invalid strategy ID (0-5)\n");
        printf("  0 - pixelwise\n");
        printf("  1 - by rows\n");
        printf("  2 - by cols\n");
        printf("  3 - blocks 32x32\n");
        printf("  4 - blocks 64x64\n");
        printf("  5 - blocks 128x128\n");
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

    printf("Applying filter %d with strategy %d (%d timed runs after warm-up)...\n", filterId, strategyId, repeat);

    Filter filter = filter_by_id(filterId);

    double min_ms, mean_ms, median_ms;
    benchmark_filter(allStrategies[strategyId + 1].fn, image, result, &filter, repeat, &min_ms, &mean_ms, &median_ms);

    printf("Saving to: '%s'\n", save_path);
    cvSaveImage(save_path, result, 0);
    printf("Time spent applying the filter: min=%8.2f ms  mean=%8.2f ms  median=%8.2f ms\n",
           min_ms, mean_ms, median_ms);

    cvReleaseImage(&image);
    cvReleaseImage(&result);
    filter_free(&filter);

    return 0;
}
