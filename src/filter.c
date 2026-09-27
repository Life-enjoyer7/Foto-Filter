#include "filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

Filter filter_create(int w, int h, const double *data, double f, double b) {
  Filter flt;
  flt.width = w;
  flt.height = h;
  flt.factor = f;
  flt.bias = b;

  flt.matrix = malloc(h * sizeof(double *));
  for (int i = 0; i < h; i++) {
    flt.matrix[i] = malloc(w * sizeof(double));
    for (int j = 0; j < w; j++) {
      flt.matrix[i][j] = data[i * w + j];
    }
  }

  return flt;
}

void filter_free(Filter *f) {
  if (f->matrix) {
    for (int i = 0; i < f->height; i++) {
      free(f->matrix[i]);
    }
    free(f->matrix);
    f->matrix = NULL;
  }
}

Filter filter_identity(void) {
  const double kernel[3][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
  return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_blur3x3(void) {
  const double kernel[3][3] = {
      {0.0, 0.2, 0.0}, {0.2, 0.2, 0.2}, {0.0, 0.2, 0.0}};
  return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_blur5x5(void) {
  const double kernel[5][5] = {{0, 0, 1, 0, 0},
                               {0, 1, 1, 1, 0},
                               {1, 1, 1, 1, 1},
                               {0, 1, 1, 1, 0},
                               {0, 0, 1, 0, 0}};
  return filter_create(5, 5, &kernel[0][0], 1.0 / 13.0, 0.0);
}

Filter filter_gaussian3x3(void) {
  const double kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  return filter_create(3, 3, &kernel[0][0], 1.0 / 16.0, 0.0);
}

Filter filter_gaussian5x5(void) {
  const double kernel[5][5] = {{1, 4, 6, 4, 1},
                               {4, 16, 24, 16, 4},
                               {6, 24, 36, 24, 6},
                               {4, 16, 24, 16, 4},
                               {1, 4, 6, 4, 1}};
  return filter_create(5, 5, &kernel[0][0], 1.0 / 256.0, 0.0);
}

Filter filter_motionblur(void) {
  const double kernel[9][9] = {
      {1, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 1, 0, 0, 0, 0, 0},
      {0, 0, 1, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 1, 0},
      {0, 0, 0, 0, 0, 0, 1, 0, 1}};
  return filter_create(9, 9, &kernel[0][0], 1.0 / 9.0, 0.0);
}

Filter filter_findedges1(void) {
  const double kernel[5][5] = {{0, 0, -1, 0, 0},
                               {0, 0, -1, 0, 0},
                               {0, 0, 2, 0, 0},
                               {0, 0, 0, 0, 0},
                               {0, 0, 0, 0, 0}};
  return filter_create(5, 5, &kernel[0][0], 1.0, 0.0);
}

Filter filter_findedges2(void) {
  const double kernel[5][5] = {{0, 0, -1, 0, 0},
                               {0, 0, -1, 0, 0},
                               {0, 0, 4, 0, 0},
                               {0, 0, -1, 0, 0},
                               {0, 0, -1, 0, 0}};
  return filter_create(5, 5, &kernel[0][0], 1.0, 0.0);
}

Filter filter_findedges3(void) {
  const double kernel[5][5] = {{-1, 0, 0, 0, 0},
                               {0, -2, 0, 0, 0},
                               {0, 0, 6, 0, 0},
                               {0, 0, 0, -2, 0},
                               {0, 0, 0, 0, -1}};
  return filter_create(5, 5, &kernel[0][0], 1.0, 0.0);
}

Filter filter_findedges4(void) {
  const double kernel[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
  return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_sharpen1(void) {
  const double kernel[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};
  return filter_create(3, 3, &kernel[0][0], 1.0, 0.0);
}

Filter filter_sharpen2(void) {
  const double kernel[5][5] = {{-1, -1, -1, -1, -1},
                               {-1, 2, 2, 2, -1},
                               {-1, 2, 8, 2, -1},
                               {-1, 2, 2, 2, -1},
                               {-1, -1, -1, -1, -1}};
  return filter_create(5, 5, &kernel[0][0], 1.0 / 8.0, 0.0);
}

Filter filter_sharpen3(void) {
  const double kernel[3][3] = {{1, 1, 1}, {1, -7, 1}, {1, 1, 1}};
  return filter_create(3, 3, &kernel[0][0], 1.0 / 8.0, 0.0);
}

Filter filter_emboss1(void) {
  const double kernel[3][3] = {{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
  return filter_create(3, 3, &kernel[0][0], 1.0, 128.0);
}

Filter filter_emboss2(void) {
  const double kernel[5][5] = {{-1, -1, -1, -1, 0},
                               {-1, -1, -1, 0, 1},
                               {-1, -1, 0, 1, 1},
                               {-1, 0, 1, 1, 1},
                               {0, 1, 1, 1, 1}};
  return filter_create(5, 5, &kernel[0][0], 1.0, 128.0);
}

typedef Filter (*filter_factory_t)(void);

const filter_factory_t filter_factories[NUM_FILTERS] = {
    filter_blur3x3,     filter_blur5x5,    filter_gaussian3x3,
    filter_gaussian5x5, filter_motionblur, filter_findedges1,
    filter_findedges2,  filter_findedges3, filter_findedges4,
    filter_sharpen1,    filter_sharpen2,   filter_sharpen3,
    filter_emboss1,     filter_emboss2,    filter_identity};

const char *const filter_names[NUM_FILTERS] = {
    "blur3x3",    "blur5x5",    "gaussian3x3", "gaussian5x5", "motionblur",
    "findedges1", "findedges2", "findedges3",  "findedges4",  "sharpen1",
    "sharpen2",   "sharpen3",   "emboss1",     "emboss2",     "identity"};

Filter filter_by_id(int id) { return filter_factories[id](); }

const char *filter_name(int id) { return filter_names[id]; }

void applyFilter(const IplImage *src, IplImage *dst, const Filter *f) {
  cvZero(dst);
  int w = src->width;
  int h = src->height;
  int step = src->widthStep;
  int channels = src->nChannels;

  const unsigned char *src_data = (const unsigned char *)src->imageData;
  unsigned char *dst_data = (unsigned char *)dst->imageData;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      double red = 0.0, green = 0.0, blue = 0.0;

      for (int fy = 0; fy < f->height; fy++) {
        for (int fx = 0; fx < f->width; fx++) {
          int ix = (x - f->width / 2 + fx + w) % w;
          int iy = (y - f->height / 2 + fy + h) % h;

          const unsigned char *pixel = src_data + iy * step + ix * channels;
          blue += pixel[0] * f->matrix[fy][fx];
          green += pixel[1] * f->matrix[fy][fx];
          red += pixel[2] * f->matrix[fy][fx];
        }
      }

      int r = (int)(f->factor * red + f->bias);
      int g = (int)(f->factor * green + f->bias);
      int b = (int)(f->factor * blue + f->bias);

      r = MAX(0, MIN(255, r));
      g = MAX(0, MIN(255, g));
      b = MAX(0, MIN(255, b));

      unsigned char *out = dst_data + y * step + x * channels;
      out[0] = (unsigned char)b;
      out[1] = (unsigned char)g;
      out[2] = (unsigned char)r;
    }
  }
}