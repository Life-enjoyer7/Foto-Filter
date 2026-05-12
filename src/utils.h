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

// Показать справку по использованию программы
void print_usage(const char *prog_name);

int imagesEqual(const IplImage *a, const IplImage *b);

#endif