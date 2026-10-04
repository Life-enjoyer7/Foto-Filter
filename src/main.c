#include "filter.h"
#include "pipeline.h"
#include "utils.h"
#include <dirent.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef CV_LOAD_IMAGE_COLOR
#define CV_LOAD_IMAGE_COLOR 1
#endif

int main(int argc, char *argv[]) {
  int filter_id = -1;
  int strategy_id = -1;
  int num_workers = 4;
  const char *input_dir = NULL;
  const char *output_dir = NULL;

  static struct option long_options[] = {{"filter", required_argument, 0, 'f'},
                                         {"tactic", required_argument, 0, 't'},
                                         {"input", required_argument, 0, 'i'},
                                         {"output", required_argument, 0, 'o'},
                                         {"workers", required_argument, 0, 'w'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int c;
  int option_index = 0;

  while ((c = getopt_long(argc, argv, "f:t:i:o:w:h", long_options,
                          &option_index)) != -1) {
    switch (c) {
    case 'f':
      filter_id = atoi(optarg);
      break;
    case 't':
      strategy_id = atoi(optarg);
      break;
    case 'i':
      input_dir = optarg;
      break;
    case 'o':
      output_dir = optarg;
      break;
    case 'w':
      num_workers = atoi(optarg);
      break;
    }
  }

  if (filter_id < 0 || filter_id >= NUM_FILTERS) {
    printf("Error: Invalid filter ID (0-26)\n");
    return 1;
  }

  if (strategy_id < 0 || strategy_id >= 7) {
    printf("Error: Invalid strategy ID (0-6)\n");
    return 1;
  }

  if (!input_dir) {
    printf("Error: Input directory required (-i)\n");
    return 1;
  }

  if (!output_dir) {
    printf("Error: Output directory required (-o)\n");
    return 1;
  }

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);
  if (num_images <= 0) {
    printf("Error: No images found in %s\n", input_dir);
    return 1;
  }

  printf("Found %d images in %s\n", num_images, input_dir);

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  double total_start = get_time_ms();

  pipeline_run(input_paths, output_paths, num_images, filter_id, strategy_id,
               num_workers, 10);

  double total_end = get_time_ms();

  printf("Total processing time: %.2f ms\n", total_end - total_start);
  printf("Average time per image: %.2f ms\n",
         (total_end - total_start) / num_images);

  for (int i = 0; i < num_images; i++) {
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(input_paths);
  free(output_paths);

  return 0;
}
