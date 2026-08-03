// Сбор сырых данных производительности в CSV для make benchmark.
// Не проверяет корректность (это делают tests/tests.c) - только время.
// Переиспользует benchmark_pipeline (src/utils) и списки фильтров/стратегий
// (tests/utils_tests), чтобы не дублировать то, что уже есть в проекте.
//
// В отличие от task2 (там каждая точка замера - одна свёртка одной картинки,
// дёшево), здесь каждая точка - полный прогон pipeline_run над ВСЕЙ папкой
// images/ (15 картинок, часть - многомегапиксельные), поэтому полный перебор
// 15 фильтров x 7 стратегий x 6 конфигураций потоков (как в tests.c) вышел бы
// на часы. Поэтому по умолчанию берём 3 конфигурации потоков вместо 6 -
// остальные легко добавить обратно в THREAD_COUNTS ниже, если есть время ждать.
#include <stdio.h>
#include <stdlib.h>
#include "../src/filter.h"
#include "../src/pipeline.h"
#include "../src/utils.h"
#include "../tests/utils_tests.h"

int main(int argc, char *argv[])
{
    int repeat = 2;
    if (argc > 1)
    {
        repeat = atoi(argv[1]);
        if (repeat < 1)
        {
            fprintf(stderr, "repeat must be >= 1\n");
            return 1;
        }
    }

    const char *input_dir = "images";
    const char *output_dir = "benchmarks/generated/bench_out";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);
    if (num_images <= 0)
    {
        fprintf(stderr, "No images found in %s\n", input_dir);
        return 1;
    }

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // 15 стандартных фильтров (те же id, что в tests.c TEST1/3/4). Padded и
    // shift-фильтры сюда не берём - по стоимости они не отличаются от базовых
    // того же размера, а composed-замеры уже есть в tests.c TEST2.
    int filter_ids[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    const char *filter_names[15] = {
        "blur3x3", "blur5x5", "gaussian3x3", "gaussian5x5", "motionblur",
        "findedges1", "findedges2", "findedges3", "findedges4",
        "sharpen1", "sharpen2", "sharpen3", "emboss1", "emboss2", "identity"};

    int thread_counts[] = {1, 4, 8, 16};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("filter,strategy,workers,num_images,min_ms,mean_ms,median_ms\n");

    for (int fi = 0; fi < 15; fi++)
    {
        for (int s = 0; s < 7; s++)
        {
            for (int t = 0; t < num_configs; t++)
            {
                int workers = thread_counts[t];

                double min_ms, mean_ms, median_ms;
                benchmark_pipeline(input_paths, output_paths, num_images,
                                   filter_ids[fi], s, workers, repeat,
                                   &min_ms, &mean_ms, &median_ms);

                printf("%s,%s,%d,%d,%.4f,%.4f,%.4f\n",
                       filter_names[fi], strategy_names[s], workers, num_images,
                       min_ms, mean_ms, median_ms);
                fflush(stdout);
            }
        }
        fprintf(stderr, "done: %s\n", filter_names[fi]);
    }

    for (int i = 0; i < num_images; i++)
    {
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(input_paths);
    free(output_paths);

    return 0;
}
