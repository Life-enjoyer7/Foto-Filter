#include "utils.h"
#include "pipeline.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int get_image_files(const char *dir_path, char ***out_paths) {
  DIR *dir;
  struct dirent *entry;
  char **paths = NULL;
  int count = 0;

  dir = opendir(dir_path);
  if (!dir) {
    return -1;
  }

  while ((entry = readdir(dir)) != NULL) {
    const char *name = entry->d_name;
    if (strstr(name, ".jpg") || strstr(name, ".jpeg") || strstr(name, ".png") ||
        strstr(name, ".bmp")) {

      char *full_path = malloc(strlen(dir_path) + strlen(name) + 2);
      sprintf(full_path, "%s/%s", dir_path, name);

      paths = realloc(paths, (count + 1) * sizeof(char *));
      paths[count++] = full_path;
    }
  }

  closedir(dir);
  *out_paths = paths;
  return count;
}

void generate_output_paths(const char *const *input_paths, char **output_paths,
                           int num_images, const char *output_dir) {
  for (int i = 0; i < num_images; i++) {
    const char *input = input_paths[i];
    const char *filename = strrchr(input, '/');
    if (filename) {
      filename++;
    } else {
      filename = input;
    }

    char *output = malloc(strlen(output_dir) + strlen(filename) + 2);
    sprintf(output, "%s/%s", output_dir, filename);
    output_paths[i] = output;
  }
}

int imagesEqual(const IplImage *a, const IplImage *b) {
  if (a->width != b->width || a->height != b->height ||
      a->nChannels != b->nChannels)
    return 0;

  int step = a->widthStep;
  int channels = a->nChannels;
  const unsigned char *data_a = (const unsigned char *)a->imageData;
  const unsigned char *data_b = (const unsigned char *)b->imageData;

  for (int y = 0; y < a->height; y++) {
    for (int x = 0; x < a->width; x++) {
      const unsigned char *pa = data_a + y * step + x * channels;
      const unsigned char *pb = data_b + y * step + x * channels;
      if (pa[0] != pb[0] || pa[1] != pb[1] || pa[2] != pb[2])
        return 0;
    }
  }
  return 1;
}

static int compare_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

void benchmark_pipeline(const char *const *input_paths,
                        const char *const *output_paths, int num_images,
                        int filter_id, int strategy_id, int num_workers,
                        int queue_capacity, int repeat, double *out_min,
                        double *out_mean, double *out_median) {
  pipeline_run(input_paths, output_paths, num_images, filter_id, strategy_id,
               num_workers, queue_capacity);

  double *samples = malloc(repeat * sizeof(double));
  double sum = 0.0;
  for (int i = 0; i < repeat; i++) {
    double start = get_time_ms();
    pipeline_run(input_paths, output_paths, num_images, filter_id, strategy_id,
                 num_workers, queue_capacity);
    double end = get_time_ms();
    samples[i] = end - start;
    sum += samples[i];
  }

  qsort(samples, repeat, sizeof(double), compare_double);

  *out_min = samples[0];
  *out_mean = sum / repeat;
  *out_median = (repeat % 2 == 0)
                    ? (samples[repeat / 2 - 1] + samples[repeat / 2]) / 2.0
                    : samples[repeat / 2];

  free(samples);
}

void sequential_run(const char *const *input_paths,
                    const char *const *output_paths, int num_images,
                    int filter_id) {
  Filter f = filter_by_id(filter_id);

  for (int i = 0; i < num_images; i++) {
    IplImage *img = cvLoadImage(input_paths[i], CV_LOAD_IMAGE_COLOR);
    if (!img) {
      fprintf(stderr, "sequential_run: failed to load %s\n", input_paths[i]);
      continue;
    }

    IplImage *dst = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    applyFilter(img, dst, &f);
    cvSaveImage(output_paths[i], dst, 0);

    cvReleaseImage(&img);
    cvReleaseImage(&dst);
  }

  filter_free(&f);
}

void benchmark_sequential(const char *const *input_paths,
                          const char *const *output_paths, int num_images,
                          int filter_id, int repeat, double *out_min,
                          double *out_mean, double *out_median) {
  sequential_run(input_paths, output_paths, num_images, filter_id);

  double *samples = malloc(repeat * sizeof(double));
  double sum = 0.0;
  for (int i = 0; i < repeat; i++) {
    double start = get_time_ms();
    sequential_run(input_paths, output_paths, num_images, filter_id);
    double end = get_time_ms();
    samples[i] = end - start;
    sum += samples[i];
  }

  qsort(samples, repeat, sizeof(double), compare_double);

  *out_min = samples[0];
  *out_mean = sum / repeat;
  *out_median = (repeat % 2 == 0)
                    ? (samples[repeat / 2 - 1] + samples[repeat / 2]) / 2.0
                    : samples[repeat / 2];

  free(samples);
}
