#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core_c.h>

// Получить текущее время в миллисекундах
double get_time_ms(void);

// Получить список файлов изображений из папки
// Возвращает количество файлов
int get_image_files(const char *dir_path, const char ***out_paths);

// Создать имена выходных файлов на основе входных путей и выходной директории
void generate_output_paths(const char **input_paths, const char **output_paths,
                           int num_images, const char *output_dir);

int imagesEqual(const IplImage *a, const IplImage *b);

// Один непрогретый прогон pipeline_run отбрасывается (прогрев ФС-кэша/страниц),
// затем `repeat` прогонов дают min/mean/median вместо одного шумного замера.
void benchmark_pipeline(const char **input_paths, const char **output_paths, int num_images,
                        int filter_id, int strategy_id, int num_workers, int repeat,
                        double *out_min, double *out_mean, double *out_median);

// То же самое, но для композиции из двух прогонов pipeline подряд (первый
// input_paths -> mid_paths, второй mid_paths -> mid_paths на месте) - для
// тестов вида "сдвиг вправо, затем сдвиг влево".
void benchmark_pipeline_twostage(const char **input_paths, const char **mid_paths, int num_images,
                                 int filter_id_a, int filter_id_b, int strategy_id,
                                 int num_workers, int repeat,
                                 double *out_min, double *out_mean, double *out_median);

#endif