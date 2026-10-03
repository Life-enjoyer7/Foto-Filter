#include "tests_utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void testIdentityFilter(void) {
  printf("\n");
  printf("                                        TEST 1: IDENTITY FILTER      "
         "                   \n");

  Filter identity = filter_identity();

  double times_pixel[15] = {0};
  double times_rows[15] = {0};
  double times_cols[15] = {0};
  double times_blocks32[15] = {0};
  double times_blocks64[15] = {0};
  double times_blocks128[15] = {0};
  double times_seq[15] = {0};

  int valid_count = 0;

  for (int j = 0; j < 15; j++) {
    IplImage *img = cvLoadImage(imagePaths[j], 1);
    if (!img) {
      printf("ERROR: Failed to load image\n");
      continue;
    }

    IplImage *result_seq =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_seq[valid_count] =
        timeFilterMs(applyFilter, img, result_seq, &identity);

    IplImage *result_pixel =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_pixel[valid_count] = timeFilterMs(applyFilterParallelPixelwise, img,
                                            result_pixel, &identity);

    IplImage *result_rows =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_rows[valid_count] =
        timeFilterMs(applyFilterParallelByRows, img, result_rows, &identity);

    IplImage *result_cols =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_cols[valid_count] =
        timeFilterMs(applyFilterParallelByCols, img, result_cols, &identity);

    IplImage *result_blocks32 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_blocks32[valid_count] = timeFilterMs(applyFilterParallelByBlocks32,
                                               img, result_blocks32, &identity);

    IplImage *result_blocks64 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_blocks64[valid_count] = timeFilterMs(applyFilterParallelByBlocks64,
                                               img, result_blocks64, &identity);

    IplImage *result_blocks128 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times_blocks128[valid_count] = timeFilterMs(
        applyFilterParallelByBlocks128, img, result_blocks128, &identity);

    int eq_pixel_seq = imagesEqual(result_pixel, result_seq);
    int eq_pixel_rows = imagesEqual(result_pixel, result_rows);
    int eq_pixel_cols = imagesEqual(result_pixel, result_cols);
    int eq_pixel_32 = imagesEqual(result_pixel, result_blocks32);
    int eq_pixel_64 = imagesEqual(result_pixel, result_blocks64);
    int eq_pixel_128 = imagesEqual(result_pixel, result_blocks128);

    if (!eq_pixel_seq || !eq_pixel_rows || !eq_pixel_cols || !eq_pixel_32 ||
        !eq_pixel_64 || !eq_pixel_128) {
      printf("Results do not match between parallel strategies (or the "
             "sequential reference)!\n");
    }
    CHECK(eq_pixel_seq && eq_pixel_rows && eq_pixel_cols && eq_pixel_32 &&
          eq_pixel_64 && eq_pixel_128);

    cvReleaseImage(&result_seq);
    cvReleaseImage(&result_pixel);
    cvReleaseImage(&result_rows);
    cvReleaseImage(&result_cols);
    cvReleaseImage(&result_blocks32);
    cvReleaseImage(&result_blocks64);
    cvReleaseImage(&result_blocks128);
    cvReleaseImage(&img);

    valid_count++;
  }

  printf("\n");
  printf("                                        PERFORMANCE RESULTS          "
         "               \n");

  for (int i = 0; i < valid_count; i++) {
    printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f "
           "ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for "
           "\t %-30s \n",
           times_seq[i], times_pixel[i], times_rows[i], times_cols[i],
           times_blocks32[i], times_blocks64[i], times_blocks128[i],
           imageNames[i]);
  }

  double total_seq = 0, total_pixel = 0, total_rows = 0, total_cols = 0;
  double total_blocks32 = 0, total_blocks64 = 0, total_blocks128 = 0;
  for (int i = 0; i < valid_count; i++) {
    total_seq += times_seq[i];
    total_pixel += times_pixel[i];
    total_rows += times_rows[i];
    total_cols += times_cols[i];
    total_blocks32 += times_blocks32[i];
    total_blocks64 += times_blocks64[i];
    total_blocks128 += times_blocks128[i];
  }
  printf("\n");
  printf(
      "Total: \t Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: "
      "%8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms \n",
      total_seq, total_pixel, total_rows, total_cols, total_blocks32,
      total_blocks64, total_blocks128);

  printf("\n");
  filter_free(&identity);
}

static void testShiftComposition(void) {
  printf("\n");
  printf("                                        TEST 2: SHIFT COMPOSITION    "
         "                     \n");

  double times[3][7][15] = {0};
  int valid_count = 0;

  const double kernel_right[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}};
  const double kernel_left[3][3] = {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
  const double kernel_up[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 1, 0}};
  const double kernel_down[3][3] = {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}};
  const double kernel_diag_up[3][3] = {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}};
  const double kernel_diag_down[3][3] = {{0, 0, 1}, {0, 0, 0}, {0, 0, 0}};

  Filter shiftRight = filter_create(3, 3, &kernel_right[0][0], 1.0, 0.0);
  Filter shiftLeft = filter_create(3, 3, &kernel_left[0][0], 1.0, 0.0);
  Filter shiftUp = filter_create(3, 3, &kernel_up[0][0], 1.0, 0.0);
  Filter shiftDown = filter_create(3, 3, &kernel_down[0][0], 1.0, 0.0);
  Filter shiftDiagUp = filter_create(3, 3, &kernel_diag_up[0][0], 1.0, 0.0);
  Filter shiftDiagDown = filter_create(3, 3, &kernel_diag_down[0][0], 1.0, 0.0);

  for (int i = 0; i < 15; i++) {
    IplImage *img = cvLoadImage(imagePaths[i], 1);
    if (!img) {
      printf("ERROR: Failed to load image %d\n", i);
      continue;
    }

    IplImage *temp_seq =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_seq =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][6][valid_count] =
        timeTwoFiltersMs(applyFilter, img, temp_seq, &shiftRight, temp_seq,
                         final_seq, &shiftLeft);

    IplImage *temp_pixel =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_pixel =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][0][valid_count] =
        timeTwoFiltersMs(applyFilterParallelPixelwise, img, temp_pixel,
                         &shiftRight, temp_pixel, final_pixel, &shiftLeft);

    IplImage *temp_rows =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_rows =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][1][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByRows, img, temp_rows, &shiftRight,
                         temp_rows, final_rows, &shiftLeft);

    IplImage *temp_cols =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_cols =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][2][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByCols, img, temp_cols, &shiftRight,
                         temp_cols, final_cols, &shiftLeft);

    IplImage *temp_32 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_32 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][3][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks32, img, temp_32,
                         &shiftRight, temp_32, final_32, &shiftLeft);

    IplImage *temp_64 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_64 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][4][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks64, img, temp_64,
                         &shiftRight, temp_64, final_64, &shiftLeft);

    IplImage *temp_128 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *final_128 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][5][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks128, img, temp_128,
                         &shiftRight, temp_128, final_128, &shiftLeft);

    int eq_pixel_seq = imagesEqual(final_pixel, final_seq);
    int eq_pixel_rows = imagesEqual(final_pixel, final_rows);
    int eq_pixel_cols = imagesEqual(final_pixel, final_cols);
    int eq_pixel_32 = imagesEqual(final_pixel, final_32);
    int eq_pixel_64 = imagesEqual(final_pixel, final_64);
    int eq_pixel_128 = imagesEqual(final_pixel, final_128);

    if (!eq_pixel_seq || !eq_pixel_rows || !eq_pixel_cols || !eq_pixel_32 ||
        !eq_pixel_64 || !eq_pixel_128) {
      printf(
          "Right-Left: results do not match between strategies for image %d!\n",
          i);
    }
    CHECK(eq_pixel_seq && eq_pixel_rows && eq_pixel_cols && eq_pixel_32 &&
          eq_pixel_64 && eq_pixel_128);

    int eq_identity = imagesEqual(img, final_pixel);
    if (!eq_identity)
      printf("Right-Left composition failed for image %d\n", i);
    CHECK(eq_identity);

    cvReleaseImage(&temp_seq);
    cvReleaseImage(&final_seq);
    cvReleaseImage(&temp_pixel);
    cvReleaseImage(&final_pixel);
    cvReleaseImage(&temp_rows);
    cvReleaseImage(&final_rows);
    cvReleaseImage(&temp_cols);
    cvReleaseImage(&final_cols);
    cvReleaseImage(&temp_32);
    cvReleaseImage(&final_32);
    cvReleaseImage(&temp_64);
    cvReleaseImage(&final_64);
    cvReleaseImage(&temp_128);
    cvReleaseImage(&final_128);

    temp_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][6][valid_count] = timeTwoFiltersMs(
        applyFilter, img, temp_seq, &shiftUp, temp_seq, final_seq, &shiftDown);

    temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][0][valid_count] =
        timeTwoFiltersMs(applyFilterParallelPixelwise, img, temp_pixel,
                         &shiftUp, temp_pixel, final_pixel, &shiftDown);

    temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][1][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByRows, img, temp_rows, &shiftUp,
                         temp_rows, final_rows, &shiftDown);

    temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][2][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByCols, img, temp_cols, &shiftUp,
                         temp_cols, final_cols, &shiftDown);

    temp_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][3][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks32, img, temp_32, &shiftUp,
                         temp_32, final_32, &shiftDown);

    temp_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][4][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks64, img, temp_64, &shiftUp,
                         temp_64, final_64, &shiftDown);

    temp_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][5][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks128, img, temp_128,
                         &shiftUp, temp_128, final_128, &shiftDown);

    eq_pixel_seq = imagesEqual(final_pixel, final_seq);
    eq_pixel_rows = imagesEqual(final_pixel, final_rows);
    eq_pixel_cols = imagesEqual(final_pixel, final_cols);
    eq_pixel_32 = imagesEqual(final_pixel, final_32);
    eq_pixel_64 = imagesEqual(final_pixel, final_64);
    eq_pixel_128 = imagesEqual(final_pixel, final_128);

    if (!eq_pixel_seq || !eq_pixel_rows || !eq_pixel_cols || !eq_pixel_32 ||
        !eq_pixel_64 || !eq_pixel_128) {
      printf("Up-Down: results do not match between strategies for image %d!\n",
             i);
    }
    CHECK(eq_pixel_seq && eq_pixel_rows && eq_pixel_cols && eq_pixel_32 &&
          eq_pixel_64 && eq_pixel_128);

    eq_identity = imagesEqual(img, final_pixel);
    if (!eq_identity)
      printf("Up-Down composition failed for image %d\n", i);
    CHECK(eq_identity);

    cvReleaseImage(&temp_seq);
    cvReleaseImage(&final_seq);
    cvReleaseImage(&temp_pixel);
    cvReleaseImage(&final_pixel);
    cvReleaseImage(&temp_rows);
    cvReleaseImage(&final_rows);
    cvReleaseImage(&temp_cols);
    cvReleaseImage(&final_cols);
    cvReleaseImage(&temp_32);
    cvReleaseImage(&final_32);
    cvReleaseImage(&temp_64);
    cvReleaseImage(&final_64);
    cvReleaseImage(&temp_128);
    cvReleaseImage(&final_128);

    temp_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_seq = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][6][valid_count] =
        timeTwoFiltersMs(applyFilter, img, temp_seq, &shiftDiagUp, temp_seq,
                         final_seq, &shiftDiagDown);

    temp_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_pixel = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][0][valid_count] =
        timeTwoFiltersMs(applyFilterParallelPixelwise, img, temp_pixel,
                         &shiftDiagUp, temp_pixel, final_pixel, &shiftDiagDown);

    temp_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_rows = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][1][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByRows, img, temp_rows,
                         &shiftDiagUp, temp_rows, final_rows, &shiftDiagDown);

    temp_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_cols = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][2][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByCols, img, temp_cols,
                         &shiftDiagUp, temp_cols, final_cols, &shiftDiagDown);

    temp_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_32 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][3][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks32, img, temp_32,
                         &shiftDiagUp, temp_32, final_32, &shiftDiagDown);

    temp_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_64 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][4][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks64, img, temp_64,
                         &shiftDiagUp, temp_64, final_64, &shiftDiagDown);

    temp_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    final_128 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][5][valid_count] =
        timeTwoFiltersMs(applyFilterParallelByBlocks128, img, temp_128,
                         &shiftDiagUp, temp_128, final_128, &shiftDiagDown);

    eq_pixel_seq = imagesEqual(final_pixel, final_seq);
    eq_pixel_rows = imagesEqual(final_pixel, final_rows);
    eq_pixel_cols = imagesEqual(final_pixel, final_cols);
    eq_pixel_32 = imagesEqual(final_pixel, final_32);
    eq_pixel_64 = imagesEqual(final_pixel, final_64);
    eq_pixel_128 = imagesEqual(final_pixel, final_128);

    if (!eq_pixel_seq || !eq_pixel_rows || !eq_pixel_cols || !eq_pixel_32 ||
        !eq_pixel_64 || !eq_pixel_128) {
      printf("Diag: results do not match between strategies for image %d!\n",
             i);
    }
    CHECK(eq_pixel_seq && eq_pixel_rows && eq_pixel_cols && eq_pixel_32 &&
          eq_pixel_64 && eq_pixel_128);

    eq_identity = imagesEqual(img, final_pixel);
    if (!eq_identity)
      printf("Diag composition failed for image %d\n", i);
    CHECK(eq_identity);

    cvReleaseImage(&temp_seq);
    cvReleaseImage(&final_seq);
    cvReleaseImage(&temp_pixel);
    cvReleaseImage(&final_pixel);
    cvReleaseImage(&temp_rows);
    cvReleaseImage(&final_rows);
    cvReleaseImage(&temp_cols);
    cvReleaseImage(&final_cols);
    cvReleaseImage(&temp_32);
    cvReleaseImage(&final_32);
    cvReleaseImage(&temp_64);
    cvReleaseImage(&final_64);
    cvReleaseImage(&temp_128);
    cvReleaseImage(&final_128);

    cvReleaseImage(&img);
    valid_count++;
  }

  filter_free(&shiftRight);
  filter_free(&shiftLeft);
  filter_free(&shiftUp);
  filter_free(&shiftDown);
  filter_free(&shiftDiagUp);
  filter_free(&shiftDiagDown);

  printf("\n                                        SHIFT COMPOSITION "
         "RESULTS\n\n");
  printf("LEFT-RIGHT:\n");
  for (int i = 0; i < valid_count; i++) {
    printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f "
           "ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for "
           "\t %-30s \n",
           times[0][6][i], times[0][0][i], times[0][1][i], times[0][2][i],
           times[0][3][i], times[0][4][i], times[0][5][i], imageNames[i]);
  }
  printf("\nUP-DOWN:\n");
  for (int i = 0; i < valid_count; i++) {
    printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f "
           "ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for "
           "\t %-30s \n",
           times[1][6][i], times[1][0][i], times[1][1][i], times[1][2][i],
           times[1][3][i], times[1][4][i], times[1][5][i], imageNames[i]);
  }
  printf("\nDIAG:\n");
  for (int i = 0; i < valid_count; i++) {
    printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f "
           "ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for "
           "\t %-30s \n",
           times[2][6][i], times[2][0][i], times[2][1][i], times[2][2][i],
           times[2][3][i], times[2][4][i], times[2][5][i], imageNames[i]);
  }
}

static void testZeroPadding(void) {
  printf("\n");
  printf("                                        TEST 3: ZERO PADDING         "
         "                                \n");
  printf("\n");

  double times[5][7][15] = {0};
  int valid_count = 0;

  Filter original_filters[5];
  original_filters[0] = filter_blur3x3();
  original_filters[1] = filter_gaussian3x3();
  original_filters[2] = filter_findedges1();
  original_filters[3] = filter_sharpen1();
  original_filters[4] = filter_emboss1();

  const double kernel_blur_padded[5][5] = {{0, 0, 0, 0, 0},
                                           {0, 0, 0.2, 0, 0},
                                           {0, 0.2, 0.2, 0.2, 0},
                                           {0, 0, 0.2, 0, 0},
                                           {0, 0, 0, 0, 0}};

  const double kernel_gauss_padded[5][5] = {{0, 0, 0, 0, 0},
                                            {0, 1, 2, 1, 0},
                                            {0, 2, 4, 2, 0},
                                            {0, 1, 2, 1, 0},
                                            {0, 0, 0, 0, 0}};

  const double kernel_edges_padded[7][7] = {
      {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0, 0}, {0, 0, 0, -1, 0, 0, 0},
      {0, 0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0},  {0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0}};

  const double kernel_sharpen_padded[5][5] = {{0, 0, 0, 0, 0},
                                              {0, -1, -1, -1, 0},
                                              {0, -1, 9, -1, 0},
                                              {0, -1, -1, -1, 0},
                                              {0, 0, 0, 0, 0}};

  const double kernel_emboss_padded[5][5] = {{0, 0, 0, 0, 0},
                                             {0, -1, -1, 0, 0},
                                             {0, -1, 0, 1, 0},
                                             {0, 0, 1, 1, 0},
                                             {0, 0, 0, 0, 0}};

  Filter padded_filters[5];
  padded_filters[0] = filter_create(5, 5, &kernel_blur_padded[0][0], 1.0, 0.0);
  padded_filters[1] =
      filter_create(5, 5, &kernel_gauss_padded[0][0], 1.0 / 16.0, 0.0);
  padded_filters[2] = filter_create(7, 7, &kernel_edges_padded[0][0], 1.0, 0.0);
  padded_filters[3] =
      filter_create(5, 5, &kernel_sharpen_padded[0][0], 1.0, 0.0);
  padded_filters[4] =
      filter_create(5, 5, &kernel_emboss_padded[0][0], 1.0, 128.0);

  const char *filter_names[5] = {"blur3x3", "gaussian3x3", "findedges1",
                                 "sharpen1", "emboss1"};

  for (int i = 0; i < 15; i++) {
    IplImage *img = cvLoadImage(imagePaths[i], 1);
    if (!img) {
      printf("ERROR: Failed to load image %d\n", i);
      continue;
    }

    for (int j = 0; j < 5; j++) {

      IplImage *result_seq =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][6][valid_count] =
          timeFilterMs(applyFilter, img, result_seq, &padded_filters[j]);

      IplImage *result_pixel =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][0][valid_count] = timeFilterMs(applyFilterParallelPixelwise, img,
                                              result_pixel, &padded_filters[j]);

      IplImage *result_rows =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][1][valid_count] = timeFilterMs(applyFilterParallelByRows, img,
                                              result_rows, &padded_filters[j]);

      IplImage *result_cols =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][2][valid_count] = timeFilterMs(applyFilterParallelByCols, img,
                                              result_cols, &padded_filters[j]);

      IplImage *result_32 =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][3][valid_count] = timeFilterMs(
          applyFilterParallelByBlocks32, img, result_32, &padded_filters[j]);

      IplImage *result_64 =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][4][valid_count] = timeFilterMs(
          applyFilterParallelByBlocks64, img, result_64, &padded_filters[j]);

      IplImage *result_128 =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      times[j][5][valid_count] = timeFilterMs(
          applyFilterParallelByBlocks128, img, result_128, &padded_filters[j]);

      int eq = imagesEqual(result_pixel, result_seq);
      if (eq)
        eq = imagesEqual(result_pixel, result_rows);
      if (eq)
        eq = imagesEqual(result_pixel, result_cols);
      if (eq)
        eq = imagesEqual(result_pixel, result_32);
      if (eq)
        eq = imagesEqual(result_pixel, result_64);
      if (eq)
        eq = imagesEqual(result_pixel, result_128);

      if (!eq) {
        printf("ERROR: Filter %s on image %d: results do not match!\n",
               filter_names[j], i);
      }
      CHECK(eq);

      cvReleaseImage(&result_seq);
      cvReleaseImage(&result_pixel);
      cvReleaseImage(&result_rows);
      cvReleaseImage(&result_cols);
      cvReleaseImage(&result_32);
      cvReleaseImage(&result_64);
      cvReleaseImage(&result_128);
    }

    cvReleaseImage(&img);
    valid_count++;
  }

  for (int j = 0; j < 5; j++) {
    filter_free(&padded_filters[j]);
    filter_free(&original_filters[j]);
  }

  printf("\n                                        ZERO PADDING RESULTS\n\n");
  for (int j = 0; j < 5; j++) {
    printf("%s:\n", filter_names[j]);
    for (int i = 0; i < valid_count; i++) {
      printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: "
             "%8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f "
             "ms for \t %-30s \n",
             times[j][6][i], times[j][0][i], times[j][1][i], times[j][2][i],
             times[j][3][i], times[j][4][i], times[j][5][i], imageNames[i]);
    }
    printf("\n");
  }
}

static void testZeroFilter(void) {
  printf("\n");
  printf("                                        TEST 4: ZERO FILTER          "
         "                               \n");
  printf("\n");

  double times[7][15] = {0};
  int valid_count = 0;

  const double kernel_zero[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  Filter zero_filter = filter_create(3, 3, &kernel_zero[0][0], 1.0, 0.0);

  for (int i = 0; i < 15; i++) {
    IplImage *img = cvLoadImage(imagePaths[i], 1);
    if (!img) {
      printf("ERROR: Failed to load image %d\n", i);
      continue;
    }

    IplImage *result_seq =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[6][valid_count] =
        timeFilterMs(applyFilter, img, result_seq, &zero_filter);

    IplImage *result_pixel =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[0][valid_count] = timeFilterMs(applyFilterParallelPixelwise, img,
                                         result_pixel, &zero_filter);

    IplImage *result_rows =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[1][valid_count] =
        timeFilterMs(applyFilterParallelByRows, img, result_rows, &zero_filter);

    IplImage *result_cols =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[2][valid_count] =
        timeFilterMs(applyFilterParallelByCols, img, result_cols, &zero_filter);

    IplImage *result_32 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[3][valid_count] = timeFilterMs(applyFilterParallelByBlocks32, img,
                                         result_32, &zero_filter);

    IplImage *result_64 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[4][valid_count] = timeFilterMs(applyFilterParallelByBlocks64, img,
                                         result_64, &zero_filter);

    IplImage *result_128 =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    times[5][valid_count] = timeFilterMs(applyFilterParallelByBlocks128, img,
                                         result_128, &zero_filter);

    int black_seq = imageIsBlack(result_seq);
    int black_pixel = imageIsBlack(result_pixel);
    int black_rows = imageIsBlack(result_rows);
    int black_cols = imageIsBlack(result_cols);
    int black_32 = imageIsBlack(result_32);
    int black_64 = imageIsBlack(result_64);
    int black_128 = imageIsBlack(result_128);

    if (!black_seq || !black_pixel || !black_rows || !black_cols || !black_32 ||
        !black_64 || !black_128) {
      printf("ERROR: Zero filter on image %d: result is not all black (seq=%d "
             "pix=%d rows=%d cols=%d blk32=%d blk64=%d blk128=%d)!\n",
             i, black_seq, black_pixel, black_rows, black_cols, black_32,
             black_64, black_128);
    }
    CHECK(black_seq && black_pixel && black_rows && black_cols && black_32 &&
          black_64 && black_128);

    cvReleaseImage(&result_seq);
    cvReleaseImage(&result_pixel);
    cvReleaseImage(&result_rows);
    cvReleaseImage(&result_cols);
    cvReleaseImage(&result_32);
    cvReleaseImage(&result_64);
    cvReleaseImage(&result_128);
    cvReleaseImage(&img);

    valid_count++;
  }

  filter_free(&zero_filter);

  printf("\n                                        ZERO FILTER RESULTS\n\n");
  for (int i = 0; i < valid_count; i++) {
    printf("│ Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: %8.2f "
           "ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms for "
           "\t %-30s \n",
           times[6][i], times[0][i], times[1][i], times[2][i], times[3][i],
           times[4][i], times[5][i], imageNames[i]);
  }

  double total_seq = 0, total_pixel = 0, total_rows = 0, total_cols = 0;
  double total_32 = 0, total_64 = 0, total_128 = 0;
  for (int i = 0; i < valid_count; i++) {
    total_seq += times[6][i];
    total_pixel += times[0][i];
    total_rows += times[1][i];
    total_cols += times[2][i];
    total_32 += times[3][i];
    total_64 += times[4][i];
    total_128 += times[5][i];
  }
  printf("\n");
  printf(
      "Total: \t Seq: %8.2f ms \t Pix: %8.2f ms \t Rows: %8.2f ms \t Cols: "
      "%8.2f ms \t Blk32: %8.2f ms \t Blk64: %8.2f ms \t Blk128: %8.2f ms \n",
      total_seq, total_pixel, total_rows, total_cols, total_32, total_64,
      total_128);
}

static void testRandomizedProperties(void) {
  printf("\n");
  printf("                                        TEST 5: RANDOMIZED "
         "PROPERTIES (random image/filter sizes)\n");

  unsigned int seed = (unsigned int)time(NULL);
  printf("  random seed: %u \n", seed);
  srand(seed);

  const int TRIALS = 30;

  const int COMPOSITION_TOLERANCE = 2;

  Filter identity = filter_identity();
  int max_diff_seen = 0;
  int parallel_mismatches = 0;

  for (int t = 0; t < TRIALS; t++) {
    int w = randomImageDim();
    int h = randomImageDim();
    IplImage *img = createRandomImage(w, h);

    for (int s = 0; s < NUM_STRATEGIES; s++) {
      IplImage *idOut =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      double tId = timeFilterMs(allStrategies[s].fn, img, idOut, &identity);
      printf("  trial %2d | img %2dx%-2d | %-9s %-10s %2dx%-2d | %9.4f ms\n", t,
             w, h, "identity", allStrategies[s].name, 3, 3, tId);
      int eqId = imagesEqual(img, idOut);
      if (!eqId)
        printf(
            "  MISMATCH: trial %d, strategy %s, identity filter, img %dx%d\n",
            t, allStrategies[s].name, w, h);
      CHECK(eqId);
      cvReleaseImage(&idOut);
    }

    int zw = randomOddSize(w);
    int zh = randomOddSize(h);
    double *zeroData = calloc(zw * zh, sizeof(double));
    Filter zeroFilter = filter_create(zw, zh, zeroData, 1.0, 0.0);
    free(zeroData);

    for (int s = 0; s < NUM_STRATEGIES; s++) {
      IplImage *zeroOut =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      double tZero =
          timeFilterMs(allStrategies[s].fn, img, zeroOut, &zeroFilter);
      printf("  trial %2d | img %2dx%-2d | %-9s %-10s %2dx%-2d | %9.4f ms\n", t,
             w, h, "zero", allStrategies[s].name, zw, zh, tZero);
      int isBlack = imageIsBlack(zeroOut);
      if (!isBlack)
        printf("  MISMATCH: trial %d, strategy %s, zero filter not all black, "
               "img %dx%d\n",
               t, allStrategies[s].name, w, h);
      CHECK(isBlack);
      cvReleaseImage(&zeroOut);
    }
    filter_free(&zeroFilter);

    int pw = randomOddSize(w);
    int ph = randomOddSize(h);
    double *baseData = malloc(pw * ph * sizeof(double));
    for (int i = 0; i < pw * ph; i++)
      baseData[i] = (double)(rand() % 5) - 2;
    Filter base = filter_create(pw, ph, baseData, 1.0, 0.0);
    free(baseData);

    int padW = pw + 2, padH = ph + 2;
    double *paddedData = calloc(padW * padH, sizeof(double));
    for (int y = 0; y < ph; y++)
      for (int x = 0; x < pw; x++)
        paddedData[(y + 1) * padW + (x + 1)] = base.matrix[y][x];
    Filter padded = filter_create(padW, padH, paddedData, 1.0, 0.0);
    free(paddedData);

    for (int s = 0; s < NUM_STRATEGIES; s++) {
      IplImage *baseOut =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      IplImage *paddedOut =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      double tPad = timeTwoFiltersMs(allStrategies[s].fn, img, baseOut, &base,
                                     img, paddedOut, &padded);
      printf("  trial %2d | img %2dx%-2d | %-9s %-10s %2dx%-2d -> %2dx%-2d | "
             "%9.4f ms\n",
             t, w, h, "pad", allStrategies[s].name, pw, ph, padW, padH, tPad);
      int eqPad = imagesEqual(baseOut, paddedOut);
      if (!eqPad)
        printf("  MISMATCH: trial %d, strategy %s, zero-padding, img %dx%d\n",
               t, allStrategies[s].name, w, h);
      CHECK(eqPad);
      cvReleaseImage(&baseOut);
      cvReleaseImage(&paddedOut);
    }
    filter_free(&base);
    filter_free(&padded);

    int aw = randomOddSize(w), ah = randomOddSize(h);
    int bw = randomOddSize(w), bh = randomOddSize(h);
    Filter fa = randomConvexFilter(aw, ah);
    Filter fb = randomConvexFilter(bw, bh);
    Filter composed = composeKernels(&fa, &fb);

    for (int s = 0; s < NUM_STRATEGIES; s++) {
      IplImage *mid = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      IplImage *twoStage =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      IplImage *oneShot =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

      double tTwoStage = timeTwoFiltersMs(allStrategies[s].fn, img, mid, &fa,
                                          mid, twoStage, &fb);
      printf("  trial %2d | img %2dx%-2d | %-9s %-10s A: %2dx%-2d + B: "
             "%2dx%-2d | %9.4f ms\n",
             t, w, h, "seq", allStrategies[s].name, aw, ah, bw, bh, tTwoStage);

      double tComposed =
          timeFilterMs(allStrategies[s].fn, img, oneShot, &composed);
      printf("  trial %2d | img %2dx%-2d | %-9s %-10s %2dx%-2d          | "
             "%9.4f ms\n",
             t, w, h, "composed", allStrategies[s].name, composed.width,
             composed.height, tComposed);

      int diff = 0;
      int ok =
          imagesApproxEqual(twoStage, oneShot, COMPOSITION_TOLERANCE, &diff);
      if (diff > max_diff_seen)
        max_diff_seen = diff;
      if (!ok) {
        printf("  MISMATCH: trial %d, strategy %s, img %dx%d, A %dx%d, B "
               "%dx%d, max diff %d\n",
               t, allStrategies[s].name, w, h, aw, ah, bw, bh, diff);
        fflush(stdout);
      }
      CHECK(ok);

      cvReleaseImage(&mid);
      cvReleaseImage(&twoStage);
      cvReleaseImage(&oneShot);
    }
    filter_free(&fa);
    filter_free(&fb);
    filter_free(&composed);

    int fw = randomOddSize(w), fh = randomOddSize(h);
    Filter randFilter = randomConvexFilter(fw, fh);

    IplImage *seqOut =
        cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    applyFilter(img, seqOut, &randFilter);

    for (int s = 1; s < NUM_STRATEGIES; s++) {
      IplImage *parOut =
          cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
      double tPar = timeFilterMs(allStrategies[s].fn, img, parOut, &randFilter);
      printf("  trial %2d | img %2dx%-2d | %-9s %2dx%-2d          | %9.4f ms\n",
             t, w, h, allStrategies[s].name, fw, fh, tPar);

      int eqPar = imagesEqual(seqOut, parOut);
      if (!eqPar) {
        parallel_mismatches++;
        printf("  MISMATCH: trial %d, strategy %s vs sequential, filter %dx%d, "
               "img %dx%d\n",
               t, allStrategies[s].name, fw, fh, w, h);
        fflush(stdout);
      }
      CHECK(eqPar);

      cvReleaseImage(&parOut);
    }

    cvReleaseImage(&seqOut);
    filter_free(&randFilter);

    cvReleaseImage(&img);
  }

  filter_free(&identity);

  printf("  max observed diff between sequential and composed-kernel pass: %d "
         "(tolerance %d)\n",
         max_diff_seen, COMPOSITION_TOLERANCE);
  printf("  parallel-vs-sequential mismatches: %d\n", parallel_mismatches);
}

static void testReferenceLibrary(void) {
  printf("\n");
  printf("                                        TEST 6: REFERENCE LIBRARY "
         "(OpenCV filter2D)\n");

  const int imageIdx[] = {6, 3, 9, 5, 8};
  const int numImages = 5;

  const int TOLERANCE = 1;
  int max_diff_seen = 0;
  int mismatches = 0;

  for (int i = 0; i < numImages; i++) {
    int idx = imageIdx[i];
    IplImage *img = cvLoadImage(imagePaths[idx], 1);
    if (!img) {
      printf("ERROR: Failed to load image %s\n", imagePaths[idx]);
      continue;
    }

    for (int j = 0; j < NUM_FILTERS; j++) {
      Filter f = filter_by_id(j);

      IplImage *reference = referenceApplyFilter(img, &f);

      for (int s = 0; s < NUM_STRATEGIES; s++) {
        IplImage *ours =
            cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double t = timeFilterMs(allStrategies[s].fn, img, ours, &f);

        int diff = 0;
        int ok = imagesApproxEqual(ours, reference, TOLERANCE, &diff);
        if (diff > max_diff_seen)
          max_diff_seen = diff;
        if (!ok) {
          mismatches++;
          printf("  MISMATCH: %s / %s / %s, max diff %d\n", imageNames[idx],
                 filter_name(j), allStrategies[s].name, diff);
        }
        CHECK(ok);

        printf("  %-22s | %-12s %2dx%-2d | %-10s | img %4dx%-4d | %8.4f ms\n",
               imageNames[idx], filter_name(j), f.width, f.height,
               allStrategies[s].name, img->width, img->height, t);

        cvReleaseImage(&ours);
      }

      cvReleaseImage(&reference);
      filter_free(&f);
    }

    cvReleaseImage(&img);
    printf("  %s vs cvFilter2D: OK (all %d filters)\n\n", imageNames[idx],
           NUM_FILTERS);
  }

  printf("  max observed diff vs OpenCV: %d (tolerance %d), mismatches: %d\n",
         max_diff_seen, TOLERANCE, mismatches);
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
