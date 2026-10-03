

#include "../src/filter.h"
#include "../src/main_utils.h"
#include "../tests/tests_utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int repeat = 5;
  if (argc > 1) {
    repeat = atoi(argv[1]);
    if (repeat < 1) {
      fprintf(stderr, "repeat must be >= 1\n");
      return 1;
    }
  }

  printf("image,width,height,filter,filter_w,filter_h,strategy,threads,repeat,"
         "min_ms,mean_ms,median_ms\n");

  int thread_counts[] = {1, 2, 4, 8, 16};
  int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

  for (int i = 0; i < 15; i++) {
    IplImage *img = cvLoadImage(imagePaths[i], 1);
    if (!img) {
      fprintf(stderr, "Warning: failed to load %s, skipping\n", imagePaths[i]);
      continue;
    }

    for (int j = 0; j < NUM_FILTERS; j++) {
      Filter f = filter_by_id(j);

      for (int s = 0; s < NUM_STRATEGIES; s++) {
        int configs = (s == 0) ? 1 : num_thread_counts;
        for (int c = 0; c < configs; c++) {
          int threads = (s == 0) ? 0 : thread_counts[c];
          set_conv_threads(threads);

          IplImage *dst =
              cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

          double min_ms, mean_ms, median_ms;
          benchmark_filter(allStrategies[s].fn, img, dst, &f, repeat, &min_ms,
                           &mean_ms, &median_ms);

          printf("%s,%d,%d,%s,%d,%d,%s,%d,%d,%.4f,%.4f,%.4f\n", imageNames[i],
                 img->width, img->height, filter_name(j), f.width, f.height,
                 allStrategies[s].name, threads, repeat, min_ms, mean_ms,
                 median_ms);
          fflush(stdout);

          cvReleaseImage(&dst);
        }
      }

      filter_free(&f);
    }

    cvReleaseImage(&img);
    fprintf(stderr, "done: %s\n", imageNames[i]);
  }

  return 0;
}
