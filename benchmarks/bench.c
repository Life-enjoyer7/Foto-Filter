#include "../src/filter.h"
#include "../src/pipeline.h"
#include "../src/utils.h"
#include "../tests/utils_tests.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int repeat = PIPELINE_BENCH_REPEAT;
  if (argc > 1) {
    repeat = atoi(argv[1]);
    if (repeat < 1) {
      fprintf(stderr, "repeat must be >= 1\n");
      return 1;
    }
  }

  const char *input_dir = "images";
  const char *output_dir = "benchmarks/generated/bench_out";

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);
  if (num_images <= 0) {
    fprintf(stderr, "No images found in %s\n", input_dir);
    return 1;
  }

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  const int num_filters = 15;

  int thread_counts[] = {1, 4, 8, 16};
  int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("filter,strategy,workers,num_images,min_ms,mean_ms,median_ms\n");

  for (int fi = 0; fi < num_filters; fi++) {
    const char *fname = filter_name(fi);
    for (int s = 0; s < NUM_STRATEGIES; s++) {
      for (int t = 0; t < num_configs; t++) {
        int workers = thread_counts[t];

        double min_ms, mean_ms, median_ms;
        benchmark_pipeline(input_paths, output_paths, num_images, fi, s,
                           workers, repeat, &min_ms, &mean_ms, &median_ms);

        printf("%s,%s,%d,%d,%.4f,%.4f,%.4f\n", fname, strategy_names[s],
               workers, num_images, min_ms, mean_ms, median_ms);
        fflush(stdout);
      }
    }
    fprintf(stderr, "done: %s\n", fname);
  }

  for (int i = 0; i < num_images; i++) {
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(input_paths);
  free(output_paths);

  return 0;
}
