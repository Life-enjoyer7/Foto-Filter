#include "../src/filter.h"
#include "../src/pipeline.h"
#include "../src/utils.h"
#include "utils_tests.h"
#include <assert.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

static void testIdentityFilter(void) {
  printf("\n");
  printf("                        TEST 1: IDENTITY FILTER                      "
         "           \n");
  printf("                   (Pipeline vs Sequential Comparison)               "
         "          \n");

  Filter identity = filter_identity();

  const char *input_dir = "images";
  const char *output_dir = "new_images";

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);

  if (num_images <= 0) {
    printf("Error: No images found in %s\n", input_dir);
    return;
  }

  printf("Found %d images in %s\n", num_images, input_dir);

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  IplImage **original_images = malloc(num_images * sizeof(IplImage *));
  for (int j = 0; j < num_images; j++) {
    original_images[j] = cvLoadImage(input_paths[j], 1);
    if (!original_images[j]) {
      printf("ERROR: Failed to load image %s\n", input_paths[j]);
    }
  }

  int thread_counts[] = {1, 2, 4, 8};
  int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

  double results[7][num_configs];

  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    int workers = thread_counts[t_idx];
    printf("\n========================================\n");
    printf("  WORKERS: %d\n", workers);
    printf("========================================\n");

    for (int s = 0; s < 7; s++) {
      printf("  %-15s : ", strategy_names[s]);
      fflush(stdout);

      double start = get_time_ms();
      pipeline_run(input_paths, output_paths, num_images, 14, s, workers, 10);
      results[s][t_idx] = get_time_ms() - start;

      for (int j = 0; j < num_images; j++) {
        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
        if (!pipeline_img)
          continue;

        int eq = imagesEqual(original_images[j], pipeline_img);
        CHECK(eq);
        cvReleaseImage(&pipeline_img);
      }
    }
  }

  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("                             SUMMARY\n");
  printf("====================================================================="
         "===========\n");
  printf("\n");

  for (int s = 0; s < 7; s++) {
    printf("  %-15s : ", strategy_names[s]);
    for (int t_idx = 0; t_idx < num_configs; t_idx++) {
      if (t_idx > 0)
        printf(" | ");
      printf("%5.0f ms", results[s][t_idx]);
    }
    printf("\n");
  }

  printf("\n");
  printf("  Workers:          ");
  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    if (t_idx > 0)
      printf(" | ");
    printf("%5d   ", thread_counts[t_idx]);
  }
  printf("\n");

  printf("\n");
  printf("  Processed images: %d\n", num_images);
  printf("\n");

  filter_free(&identity);
  for (int i = 0; i < num_images; i++) {
    if (original_images[i])
      cvReleaseImage(&original_images[i]);
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(original_images);
  free(input_paths);
  free(output_paths);
}

static void testShiftComposition(void) {
  printf("\n");
  printf("                        TEST 2: SHIFT COMPOSITION                    "
         "          \n");
  printf("                   (Pipeline Strategies Comparison)                  "
         "          \n");

  Filter shiftRight = filter_shift_right();
  Filter shiftLeft = filter_shift_left();
  Filter shiftUp = filter_shift_up();
  Filter shiftDown = filter_shift_down();
  Filter shiftDiagUp = filter_shift_diag_up();
  Filter shiftDiagDown = filter_shift_diag_down();

  const char *input_dir = "images";
  const char *output_dir = "new_images";

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);

  if (num_images <= 0) {
    printf("Error: No images found in %s\n", input_dir);
    return;
  }

  printf("Found %d images in %s\n", num_images, input_dir);

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  IplImage **original_images = malloc(num_images * sizeof(IplImage *));
  for (int j = 0; j < num_images; j++) {
    original_images[j] = cvLoadImage(input_paths[j], 1);
    if (!original_images[j]) {
      printf("ERROR: Failed to load image %s\n", input_paths[j]);
    }
  }

  int thread_counts[] = {1, 2, 4, 8};
  int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

  double results[3][7][num_configs];
  const char *comp_names[3] = {"Right-Left", "Up-Down", "Diag"};

  int filter_indices[3][2] = {{15, 16}, {17, 18}, {19, 20}};

  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    int workers = thread_counts[t_idx];
    printf("\n========================================\n");
    printf("  WORKERS: %d\n", workers);
    printf("========================================\n");

    for (int comp = 0; comp < 3; comp++) {
      printf("\n  %s composition:\n", comp_names[comp]);

      for (int s = 0; s < 7; s++) {
        printf("    %-15s : ", strategy_names[s]);
        fflush(stdout);

        double start = get_time_ms();
        pipeline_run(input_paths, output_paths, num_images,
                     filter_indices[comp][0], s, workers, 10);
        pipeline_run(output_paths, output_paths, num_images,
                     filter_indices[comp][1], s, workers, 10);
        results[comp][s][t_idx] = get_time_ms() - start;

        for (int j = 0; j < num_images; j++) {
          IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
          int eq = imagesEqual(original_images[j], pipeline_img);
          CHECK(eq);
          cvReleaseImage(&pipeline_img);
        }
      }
    }
  }

  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("                             SUMMARY\n");
  printf("====================================================================="
         "===========\n");
  printf("\n");

  for (int comp = 0; comp < 3; comp++) {
    printf("  %s composition:\n", comp_names[comp]);
    for (int s = 0; s < 7; s++) {
      printf("    %-15s : ", strategy_names[s]);
      for (int t_idx = 0; t_idx < num_configs; t_idx++) {
        if (t_idx > 0)
          printf(" | ");
        printf("%5.0f ms", results[comp][s][t_idx]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("  Workers:          ");
  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    if (t_idx > 0)
      printf(" | ");
    printf("%5d   ", thread_counts[t_idx]);
  }
  printf("\n");

  printf("\n");
  printf("  Processed images: %d\n", num_images);
  printf("\n");

  filter_free(&shiftRight);
  filter_free(&shiftLeft);
  filter_free(&shiftUp);
  filter_free(&shiftDown);
  filter_free(&shiftDiagUp);
  filter_free(&shiftDiagDown);

  for (int i = 0; i < num_images; i++) {
    if (original_images[i])
      cvReleaseImage(&original_images[i]);
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(original_images);
  free(input_paths);
  free(output_paths);
}

static void testZeroPadding(void) {
  printf("\n");
  printf("                        TEST 3: ZERO PADDING                         "
         "           \n");
  printf("                   (Pipeline Strategies Comparison)                  "
         "          \n");

  const char *input_dir = "images";
  const char *output_dir = "new_images";

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);

  if (num_images <= 0) {
    printf("Error: No images found in %s\n", input_dir);
    return;
  }

  printf("Found %d images in %s\n", num_images, input_dir);

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  int padded_indices[5] = {21, 22, 23, 24, 25};
  const char *filter_names[5] = {"blur3x3", "gaussian3x3", "findedges1",
                                 "sharpen1", "emboss1"};

  Filter original_filters[5];
  original_filters[0] = filter_blur3x3();
  original_filters[1] = filter_gaussian3x3();
  original_filters[2] = filter_findedges1();
  original_filters[3] = filter_sharpen1();
  original_filters[4] = filter_emboss1();

  IplImage **original_images = malloc(num_images * sizeof(IplImage *));
  for (int j = 0; j < num_images; j++) {
    original_images[j] = cvLoadImage(input_paths[j], 1);
    if (!original_images[j]) {
      printf("ERROR: Failed to load image %s\n", input_paths[j]);
    }
  }

  int thread_counts[] = {1, 2, 4, 8};
  int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

  double results[5][7][num_configs];

  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    int workers = thread_counts[t_idx];
    printf("\n========================================\n");
    printf("  WORKERS: %d\n", workers);
    printf("========================================\n");

    for (int f = 0; f < 5; f++) {
      printf("\n  Filter: %s\n", filter_names[f]);

      for (int s = 0; s < 7; s++) {
        printf("    %-15s : ", strategy_names[s]);
        fflush(stdout);

        double start = get_time_ms();
        pipeline_run(input_paths, output_paths, num_images, padded_indices[f],
                     s, workers, 10);
        results[f][s][t_idx] = get_time_ms() - start;

        for (int j = 0; j < num_images; j++) {
          IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
          if (!pipeline_img)
            continue;

          IplImage *original_result = cvCreateImage(
              cvGetSize(original_images[j]), original_images[j]->depth,
              original_images[j]->nChannels);
          applyFilter(original_images[j], original_result,
                      &original_filters[f]);

          int eq = imagesEqual(pipeline_img, original_result);
          CHECK(eq);

          cvReleaseImage(&pipeline_img);
          cvReleaseImage(&original_result);
        }
      }
    }
  }

  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("                             SUMMARY\n");
  printf("====================================================================="
         "===========\n");
  printf("\n");

  for (int f = 0; f < 5; f++) {
    printf("  Filter: %s\n", filter_names[f]);
    for (int s = 0; s < 7; s++) {
      printf("    %-15s : ", strategy_names[s]);
      for (int t_idx = 0; t_idx < num_configs; t_idx++) {
        if (t_idx > 0)
          printf(" | ");
        printf("%5.0f ms", results[f][s][t_idx]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("  Workers:          ");
  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    if (t_idx > 0)
      printf(" | ");
    printf("%5d   ", thread_counts[t_idx]);
  }
  printf("\n");

  printf("\n");
  printf("  Processed images: %d\n", num_images);
  printf("\n");

  for (int i = 0; i < 5; i++) {
    filter_free(&original_filters[i]);
  }
  for (int i = 0; i < num_images; i++) {
    if (original_images[i])
      cvReleaseImage(&original_images[i]);
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(original_images);
  free(input_paths);
  free(output_paths);
}

static void testZeroFilter(void) {
  printf("\n");
  printf("                        TEST 4: ZERO FILTER                          "
         "           \n");
  printf("                   (Pipeline Strategies Comparison)                  "
         "          \n");

  Filter zero = filter_zero();

  const char *input_dir = "images";
  const char *output_dir = "new_images";

  char **input_paths = NULL;
  int num_images = get_image_files(input_dir, &input_paths);

  if (num_images <= 0) {
    printf("Error: No images found in %s\n", input_dir);
    return;
  }

  printf("Found %d images in %s\n", num_images, input_dir);

  char **output_paths = malloc(num_images * sizeof(char *));
  generate_output_paths(input_paths, output_paths, num_images, output_dir);

  IplImage **original_images = malloc(num_images * sizeof(IplImage *));
  for (int j = 0; j < num_images; j++) {
    original_images[j] = cvLoadImage(input_paths[j], 1);
    if (!original_images[j]) {
      printf("ERROR: Failed to load image %s\n", input_paths[j]);
    }
  }

  int thread_counts[] = {1, 2, 4, 8};
  int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

  double results[7][num_configs];

  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    int workers = thread_counts[t_idx];
    printf("\n========================================\n");
    printf("  WORKERS: %d\n", workers);
    printf("========================================\n");

    for (int s = 0; s < 7; s++) {
      printf("  %-15s : ", strategy_names[s]);
      fflush(stdout);

      double start = get_time_ms();
      pipeline_run(input_paths, output_paths, num_images, 26, s, workers, 10);
      results[s][t_idx] = get_time_ms() - start;

      for (int j = 0; j < num_images; j++) {
        IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
        if (!pipeline_img)
          continue;

        int step = pipeline_img->widthStep;
        int channels = pipeline_img->nChannels;
        const unsigned char *data =
            (const unsigned char *)pipeline_img->imageData;
        int all_black = 1;
        for (int y = 0; y < pipeline_img->height && all_black; y++) {
          for (int x = 0; x < pipeline_img->width; x++) {
            const unsigned char *pixel = data + y * step + x * channels;
            if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0) {
              all_black = 0;
              break;
            }
          }
        }
        CHECK(all_black);
        cvReleaseImage(&pipeline_img);
      }
    }
  }

  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("                             SUMMARY\n");
  printf("====================================================================="
         "===========\n");
  printf("\n");

  for (int s = 0; s < 7; s++) {
    printf("  %-15s : ", strategy_names[s]);
    for (int t_idx = 0; t_idx < num_configs; t_idx++) {
      if (t_idx > 0)
        printf(" | ");
      printf("%5.0f ms", results[s][t_idx]);
    }
    printf("\n");
  }

  printf("\n");
  printf("  Workers:          ");
  for (int t_idx = 0; t_idx < num_configs; t_idx++) {
    if (t_idx > 0)
      printf(" | ");
    printf("%5d   ", thread_counts[t_idx]);
  }
  printf("\n");

  printf("\n");
  printf("  Processed images: %d\n", num_images);
  printf("\n");

  filter_free(&zero);
  for (int i = 0; i < num_images; i++) {
    if (original_images[i])
      cvReleaseImage(&original_images[i]);
    free(input_paths[i]);
    free(output_paths[i]);
  }
  free(original_images);
  free(input_paths);
  free(output_paths);
}

static void testRandomizedProperties(void) {
  printf("\n");
  printf("                        TEST 5: RANDOMIZED PROPERTIES                "
         "           \n");
  printf("                   (random synthetic images, pipeline)               "
         "           \n");

  unsigned int seed = (unsigned int)time(NULL);
  printf("  random seed: %u\n", seed);
  srand(seed);

  mkdir("tmp_random_in", 0755);
  mkdir("tmp_random_out", 0755);
  mkdir("tmp_random_mid", 0755);

  int mismatches = 0;

  for (int t = 0; t < RANDOM_TEST_TRIALS; t++) {
    for (int i = 0; i < MAX_TRIAL_IMAGES; i++) {
      char buf[64];
      snprintf(buf, sizeof(buf), "tmp_random_in/rand_%d.png", i);
      remove(buf);
    }

    int num_images = 1 + rand() % MAX_TRIAL_IMAGES;

    char *in_paths[MAX_TRIAL_IMAGES];
    char *out_paths[MAX_TRIAL_IMAGES];
    char *mid_paths[MAX_TRIAL_IMAGES];
    IplImage *originals[MAX_TRIAL_IMAGES];
    int minDim = 1 << 30;

    for (int i = 0; i < num_images; i++) {
      int w = randomImageDim();
      int h = randomImageDim();
      if (w < minDim)
        minDim = w;
      if (h < minDim)
        minDim = h;

      originals[i] = createRandomImage(w, h);

      char buf[64];
      snprintf(buf, sizeof(buf), "tmp_random_in/rand_%d.png", i);
      in_paths[i] = strdup(buf);
      snprintf(buf, sizeof(buf), "tmp_random_out/rand_%d.png", i);
      out_paths[i] = strdup(buf);
      snprintf(buf, sizeof(buf), "tmp_random_mid/rand_%d.png", i);
      mid_paths[i] = strdup(buf);

      cvSaveImage(in_paths[i], originals[i], 0);
    }

    int filter_id = (t == 0)   ? 14
                    : (t == 1) ? 26
                               : randomFilterIdForDim(minDim);
    int num_workers = 1 + rand() % 8;
    int strategy_id = rand() % 7;

    Filter expectedFilter = filter_by_id(filter_id);

    pipeline_run(in_paths, out_paths, num_images, filter_id, strategy_id,
                 num_workers, 10);

    for (int i = 0; i < num_images; i++) {
      IplImage *result = cvLoadImage(out_paths[i], 1);
      IplImage *expected =
          cvCreateImage(cvGetSize(originals[i]), originals[i]->depth,
                        originals[i]->nChannels);
      applyFilter(originals[i], expected, &expectedFilter);

      int eq = result && imagesEqual(result, expected);
      if (!eq) {
        mismatches++;
        printf("  MISMATCH: trial %d, strategy %s, filter_id %d, image %d "
               "(%dx%d)\n",
               t, strategy_names[strategy_id], filter_id, i,
               originals[i]->width, originals[i]->height);
        fflush(stdout);
      }
      CHECK(eq);

      if (result)
        cvReleaseImage(&result);
      cvReleaseImage(&expected);
    }

    for (int s = 0; s < 7; s++) {
      pipeline_run(in_paths, out_paths, num_images, filter_id, s, num_workers,
                   10);

      for (int i = 0; i < num_images; i++) {
        IplImage *result = cvLoadImage(out_paths[i], 1);
        IplImage *expected =
            cvCreateImage(cvGetSize(originals[i]), originals[i]->depth,
                          originals[i]->nChannels);
        applyFilter(originals[i], expected, &expectedFilter);

        int eq = result && imagesEqual(result, expected);
        if (!eq) {
          mismatches++;
          printf("  MISMATCH: trial %d, strategy %s vs sequential, filter_id "
                 "%d, image %d\n",
                 t, strategy_names[s], filter_id, i);
          fflush(stdout);
        }
        CHECK(eq);

        if (result)
          cvReleaseImage(&result);
        cvReleaseImage(&expected);
      }
    }

    int comp = rand() % 3;
    int compA = 15 + comp * 2;
    int compB = 16 + comp * 2;
    pipeline_run(in_paths, mid_paths, num_images, compA, strategy_id,
                 num_workers, 10);
    pipeline_run(mid_paths, mid_paths, num_images, compB, strategy_id,
                 num_workers, 10);

    for (int i = 0; i < num_images; i++) {
      IplImage *result = cvLoadImage(mid_paths[i], 1);
      int eq = result && imagesEqual(result, originals[i]);
      if (!eq) {
        mismatches++;
        printf("  MISMATCH: trial %d, shift composition (%d,%d), image %d\n", t,
               compA, compB, i);
        fflush(stdout);
      }
      CHECK(eq);
      if (result)
        cvReleaseImage(&result);
    }

    filter_free(&expectedFilter);
    for (int i = 0; i < num_images; i++) {
      cvReleaseImage(&originals[i]);
      free(in_paths[i]);
      free(out_paths[i]);
      free(mid_paths[i]);
    }
  }

  printf("  mismatches: %d\n", mismatches);
}

static void testReferenceLibrary(void) {
  printf("\n");
  printf("                        TEST 6: REFERENCE LIBRARY (OpenCV filter2D)  "
         "           \n");
  printf("                   (Full pipeline vs cv::filter2D)                   "
         "           \n");

  const char *curated_images[] = {
      "images/lambo_236x236.png",       "images/ferari_320x320.png",
      "images/mustang_736x736.png",     "images/ford_1080x1080.png",
      "images/maseratti_2048x2048.png",
  };
  const char *curated_names[] = {"lambo_236x236", "ferari_320x320",
                                 "mustang_736x736", "ford_1080x1080",
                                 "maseratti_2048x2048"};
  const int num_curated = 5;

  const int NUM_BASIC_FILTERS = 15;

  const int TOLERANCE = 1;
  int mismatches = 0;

  mkdir("tmp_ref_out", 0755);

  for (int i = 0; i < num_curated; i++) {
    IplImage *original = cvLoadImage(curated_images[i], 1);
    if (!original) {
      printf("ERROR: failed to load %s\n", curated_images[i]);
      continue;
    }

    for (int j = 0; j < NUM_BASIC_FILTERS; j++) {
      Filter f = filter_by_id(j);

      char out_path[128];
      snprintf(out_path, sizeof(out_path), "tmp_ref_out/%s_%s.png",
               curated_names[i], filter_name(j));

      const char *in_arr[1] = {curated_images[i]};
      const char *out_arr[1] = {out_path};

      pipeline_run(in_arr, out_arr, 1, j, 0, 2, 10);

      IplImage *ours = cvLoadImage(out_path, 1);
      IplImage *reference = referenceApplyFilter(original, &f);

      int ok = ours && imagesApproxEqual(ours, reference, TOLERANCE);
      if (!ok) {
        mismatches++;
        printf("  MISMATCH: %s / %s\n", curated_names[i], filter_name(j));
      }
      CHECK(ok);

      if (ours)
        cvReleaseImage(&ours);
      cvReleaseImage(&reference);
      filter_free(&f);
    }

    cvReleaseImage(&original);
    printf("  %s vs cv::filter2D: OK (all 15 filters)\n", curated_names[i]);
  }

  printf("  mismatches: %d\n", mismatches);
}

int main(void) {
  RUN_TEST(testIdentityFilter, "TEST 1: IDENTITY FILTER");
  RUN_TEST(testShiftComposition, "TEST 2: SHIFT COMPOSITION");
  RUN_TEST(testZeroPadding, "TEST 3: ZERO PADDING");
  RUN_TEST(testZeroFilter, "TEST 4: ZERO FILTER");
  RUN_TEST(testRandomizedProperties, "TEST 5: RANDOMIZED PROPERTIES");
  RUN_TEST(testReferenceLibrary, "TEST 6: REFERENCE LIBRARY");

  if (g_test_failures == 0) {
    printf("\n                                        ALL TESTS PASSED\n");
  } else {
    printf("\n                                        TOTAL: %d FAILURE(S)\n",
           g_test_failures);
  }

  assert(g_test_failures == 0);
  return g_test_failures == 0 ? 0 : 1;
}
