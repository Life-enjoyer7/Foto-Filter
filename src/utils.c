#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>

double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int get_image_files(const char *dir_path, const char ***out_paths)
{
    DIR *dir;
    struct dirent *entry;
    const char **paths = NULL;
    int count = 0;

    dir = opendir(dir_path);
    if (!dir)
    {
        return -1;
    }

    while ((entry = readdir(dir)) != NULL)
    {
        const char *name = entry->d_name;
        // Проверяем расширение
        if (strstr(name, ".jpg") || strstr(name, ".jpeg") ||
            strstr(name, ".png") || strstr(name, ".bmp"))
        {

            // Выделяем память под путь
            char *full_path = (char *)malloc(strlen(dir_path) + strlen(name) + 2);
            sprintf(full_path, "%s/%s", dir_path, name);

            paths = (const char **)realloc(paths, (count + 1) * sizeof(const char *));
            paths[count++] = full_path;
        }
    }

    closedir(dir);
    *out_paths = paths;
    return count;
}

void generate_output_paths(const char **input_paths, const char **output_paths,
                           int num_images, const char *output_dir)
{
    for (int i = 0; i < num_images; i++)
    {
        const char *input = input_paths[i];
        const char *filename = strrchr(input, '/');
        if (filename)
        {
            filename++; // пропускаем '/'
        }
        else
        {
            filename = input;
        }

        char *output = (char *)malloc(strlen(output_dir) + strlen(filename) + 2);
        sprintf(output, "%s/%s", output_dir, filename);
        output_paths[i] = output;
    }
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

void print_usage(const char *prog_name)
{
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("\n");
    printf("Options:\n");
    printf("  -f, --filter NUM      Filter ID (0-14)\n");
    printf("  -t, --tactic NUM      Strategy ID (0-5)\n");
    printf("  -i, --input DIR       Input directory with images\n");
    printf("  -o, --output DIR      Output directory for results\n");
    printf("  -w, --workers NUM     Number of worker threads (default: 4)\n");
    printf("  -h, --help            Show this help\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -f 2 -t 1 -i images/ -o results/ -w 4\n", prog_name);
}
