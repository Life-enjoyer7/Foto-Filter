#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core_c.h>

double get_time_ms(void);

int get_image_files(const char *dir_path, char ***out_paths);

void generate_output_paths(const char *const *input_paths, char **output_paths,
                           int num_images, const char *output_dir);

int imagesEqual(const IplImage *a, const IplImage *b);

void sequential_run(const char *const *input_paths,
                    const char *const *output_paths, int num_images,
                    int filter_id);

void benchmark_pipeline(const char *const *input_paths,
                        const char *const *output_paths, int num_images,
                        int filter_id, int strategy_id, int num_workers,
                        int queue_capacity, int repeat, double *out_min,
                        double *out_mean, double *out_median);

void benchmark_sequential(const char *const *input_paths,
                          const char *const *output_paths, int num_images,
                          int filter_id, int repeat, double *out_min,
                          double *out_mean, double *out_median);

#endif
