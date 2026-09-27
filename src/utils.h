#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core_c.h>

double get_time_ms(void);

int get_image_files(const char *dir_path, char ***out_paths);

void generate_output_paths(const char *const *input_paths, char **output_paths,
                           int num_images, const char *output_dir);

int imagesEqual(const IplImage *a, const IplImage *b);

void benchmark_pipeline(const char *const *input_paths,
                        const char *const *output_paths, int num_images,
                        int filter_id, int strategy_id, int num_workers,
                        int repeat, double *out_min, double *out_mean,
                        double *out_median);

#endif
