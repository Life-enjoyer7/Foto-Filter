#include "../src/filter.h"
#include "../src/pipeline.h"
#include "../src/utils.h"
#include "../tests/utils_tests.h"
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_QUEUE_CAPACITY 10
#define QUEUE_WORKERS 10

int main(int argc, char *argv[])
{
  int repeat = PIPELINE_BENCH_REPEAT;
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

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);
  if (num_images <= 0)
  {
    fprintf(stderr, "No images found in %s\n", input_dir);
    return 1;
  }

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  const int num_filters = 15;

  int thread_counts[] = {1, 4, 8, 16};
  int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

  int queue_capacities[] = {1, 2, 4, 8, 16, 32, 64};
  int num_queue_capacities =
      sizeof(queue_capacities) / sizeof(queue_capacities[0]);

  printf("filter,strategy,workers,queue_capacity,num_images,min_ms,mean_ms,"
         "median_ms\n");

  for (int fi = 0; fi < num_filters; fi++)
  {
    const char *fname = filter_name(fi);
    double min_ms, mean_ms, median_ms;

    benchmark_sequential(input_paths, output_paths, num_images, fi, repeat,
                         &min_ms, &mean_ms, &median_ms);
    printf("%s,%s,%d,%d,%d,%.4f,%.4f,%.4f\n", fname, "Baseline", 1, 0,
           num_images, min_ms, mean_ms, median_ms);

    for (int s = 1; s < NUM_STRATEGIES; s++)
    {
      for (int t = 0; t < num_thread_counts; t++)
      {
        int workers = thread_counts[t];
        benchmark_pipeline(input_paths, output_paths, num_images, fi, s,
                           workers, DEFAULT_QUEUE_CAPACITY, repeat, &min_ms,
                           &mean_ms, &median_ms);
        printf("%s,%s,%d,%d,%d,%.4f,%.4f,%.4f\n", fname, strategy_names[s],
               workers, DEFAULT_QUEUE_CAPACITY, num_images, min_ms, mean_ms,
               median_ms);
        fflush(stdout);
      }
    }

    for (int s = 1; s < NUM_STRATEGIES; s++)
    {
      for (int q = 0; q < num_queue_capacities; q++)
      {
        int capacity = queue_capacities[q];
        benchmark_pipeline(input_paths, output_paths, num_images, fi, s,
                           QUEUE_WORKERS, capacity, repeat, &min_ms, &mean_ms,
                           &median_ms);
        printf("%s,%s,%d,%d,%d,%.4f,%.4f,%.4f\n", fname, strategy_names[s],
               QUEUE_WORKERS, capacity, num_images, min_ms, mean_ms, median_ms);
        fflush(stdout);
      }
    }

    fprintf(stderr, "done: %s\n", fname);
  }

  for (int i = 0; i < num_images; i++)
  {
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(input_paths);
  free(output_paths);

  return 0;
}
