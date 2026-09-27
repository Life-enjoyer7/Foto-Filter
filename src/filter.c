#include "filter.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

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

typedef struct {
  const IplImage *src;
  IplImage *dst;
  const Filter *f;
  int w;
  int h;
  int startIdx;
  int endIdx;
} ThreadArgs;

static void *processPixelRange(void *args) {
  ThreadArgs *a = (ThreadArgs *)args;
  const IplImage *src = a->src;
  IplImage *dst = a->dst;
  const Filter *f = a->f;
  int w = a->w;
  int h = a->h;
  int startIdx = a->startIdx;
  int endIdx = a->endIdx;

  int step = src->widthStep;
  int channels = src->nChannels;
  const unsigned char *src_data = (const unsigned char *)src->imageData;
  unsigned char *dst_data = (unsigned char *)dst->imageData;

  for (int idx = startIdx; idx < endIdx; idx++) {
    int y = idx / w;
    int x = idx % w;

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

  return NULL;
}

void applyFilterParallelPixelwise(const IplImage *src, IplImage *dst,
                                  const Filter *f) {

  cvZero(dst);

  int w = src->width;
  int h = src->height;
  int totalPixels = w * h;

  int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
  if (numThreads <= 0)
    numThreads = 4;

  int pixelsPerThread = totalPixels / numThreads;
  int remainder = totalPixels % numThreads;

  pthread_t threads[numThreads];
  ThreadArgs args[numThreads];

  for (int t = 0; t < numThreads; t++) {
    int startIdx = t * pixelsPerThread + (t < remainder ? t : remainder);
    int endIdx = startIdx + pixelsPerThread + (t < remainder ? 1 : 0);

    args[t].src = src;
    args[t].dst = dst;
    args[t].f = f;
    args[t].w = w;
    args[t].h = h;
    args[t].startIdx = startIdx;
    args[t].endIdx = endIdx;

    pthread_create(&threads[t], NULL, processPixelRange, &args[t]);
  }

  for (int t = 0; t < numThreads; t++) {
    pthread_join(threads[t], NULL);
  }
}

typedef struct {
  const IplImage *src;
  IplImage *dst;
  const Filter *f;
  int w;
  int h;
  int startRow;
  int endRow;
} ThreadArgsRows;

static void *processRowRange(void *args) {
  ThreadArgsRows *a = (ThreadArgsRows *)args;
  const IplImage *src = a->src;
  IplImage *dst = a->dst;
  const Filter *f = a->f;
  int w = a->w;
  int h = a->h;
  int startRow = a->startRow;
  int endRow = a->endRow;

  int step = src->widthStep;
  int channels = src->nChannels;
  const unsigned char *src_data = (const unsigned char *)src->imageData;
  unsigned char *dst_data = (unsigned char *)dst->imageData;

  for (int y = startRow; y < endRow; y++) {
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

  return NULL;
}

void applyFilterParallelByRows(const IplImage *src, IplImage *dst,
                               const Filter *f) {

  cvZero(dst);

  int w = src->width;
  int h = src->height;

  int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
  if (numThreads <= 0)
    numThreads = 4;

  if (numThreads > h)
    numThreads = h;

  int rowsPerThread = h / numThreads;
  int remainder = h % numThreads;

  pthread_t threads[numThreads];
  ThreadArgsRows args[numThreads];

  for (int t = 0; t < numThreads; t++) {
    int startRow = t * rowsPerThread + (t < remainder ? t : remainder);
    int endRow = startRow + rowsPerThread + (t < remainder ? 1 : 0);

    args[t].src = src;
    args[t].dst = dst;
    args[t].f = f;
    args[t].w = w;
    args[t].h = h;
    args[t].startRow = startRow;
    args[t].endRow = endRow;

    pthread_create(&threads[t], NULL, processRowRange, &args[t]);
  }

  for (int t = 0; t < numThreads; t++) {
    pthread_join(threads[t], NULL);
  }
}

typedef struct {
  const IplImage *src;
  IplImage *dst;
  const Filter *f;
  int w;
  int h;
  int startCol;
  int endCol;
} ThreadArgsCols;

static void *processColRange(void *args) {
  ThreadArgsCols *a = (ThreadArgsCols *)args;
  const IplImage *src = a->src;
  IplImage *dst = a->dst;
  const Filter *f = a->f;
  int w = a->w;
  int h = a->h;
  int startCol = a->startCol;
  int endCol = a->endCol;

  int step = src->widthStep;
  int channels = src->nChannels;
  const unsigned char *src_data = (const unsigned char *)src->imageData;
  unsigned char *dst_data = (unsigned char *)dst->imageData;

  for (int y = 0; y < h; y++) {
    for (int x = startCol; x < endCol; x++) {
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

  return NULL;
}

void applyFilterParallelByCols(const IplImage *src, IplImage *dst,
                               const Filter *f) {

  cvZero(dst);

  int w = src->width;
  int h = src->height;

  int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
  if (numThreads <= 0)
    numThreads = 4;

  if (numThreads > w)
    numThreads = w;

  int colsPerThread = w / numThreads;
  int remainder = w % numThreads;

  pthread_t threads[numThreads];
  ThreadArgsCols args[numThreads];

  for (int t = 0; t < numThreads; t++) {
    int startCol = t * colsPerThread + (t < remainder ? t : remainder);
    int endCol = startCol + colsPerThread + (t < remainder ? 1 : 0);

    args[t].src = src;
    args[t].dst = dst;
    args[t].f = f;
    args[t].w = w;
    args[t].h = h;
    args[t].startCol = startCol;
    args[t].endCol = endCol;

    pthread_create(&threads[t], NULL, processColRange, &args[t]);
  }

  for (int t = 0; t < numThreads; t++) {
    pthread_join(threads[t], NULL);
  }
}

typedef struct {
  const IplImage *src;
  IplImage *dst;
  const Filter *f;
  int w, h;
  int blocksX;
  int blockW;
  int blockH;
  int startBlock;
  int endBlock;
} ThreadArgsBlock;

static void *processBlock(void *args) {
  ThreadArgsBlock *a = (ThreadArgsBlock *)args;
  const IplImage *src = a->src;
  IplImage *dst = a->dst;
  const Filter *f = a->f;
  int w = a->w;
  int h = a->h;
  int step = src->widthStep;
  int channels = src->nChannels;
  const unsigned char *src_data = (const unsigned char *)src->imageData;
  unsigned char *dst_data = (unsigned char *)dst->imageData;

  for (int bi = a->startBlock; bi < a->endBlock; bi++) {

    int bx = bi % a->blocksX;
    int by = bi / a->blocksX;

    int startX = bx * a->blockW;
    int startY = by * a->blockH;
    int endX = MIN(startX + a->blockW, w);
    int endY = MIN(startY + a->blockH, h);

    for (int y = startY; y < endY; y++) {
      for (int x = startX; x < endX; x++) {
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

  return NULL;
}

void applyFilterParallelByBlocks(const IplImage *src, IplImage *dst,
                                 const Filter *f, int blockW, int blockH) {
  cvZero(dst);

  int w = src->width;
  int h = src->height;

  int blocksX = (w + blockW - 1) / blockW;
  int blocksY = (h + blockH - 1) / blockH;
  int totalBlocks = blocksX * blocksY;

  int numThreads = sysconf(_SC_NPROCESSORS_ONLN);
  if (numThreads <= 0)
    numThreads = 4;
  if (numThreads > totalBlocks)
    numThreads = totalBlocks;

  pthread_t threads[numThreads];
  ThreadArgsBlock args[numThreads];

  int blocksPerThread = totalBlocks / numThreads;
  int remainder = totalBlocks % numThreads;
  int blockIdx = 0;

  for (int t = 0; t < numThreads; t++) {
    args[t].src = src;
    args[t].dst = dst;
    args[t].f = f;
    args[t].w = w;
    args[t].h = h;
    args[t].blocksX = blocksX;
    args[t].blockW = blockW;
    args[t].blockH = blockH;
    args[t].startBlock = blockIdx;
    args[t].endBlock = blockIdx + blocksPerThread + (t < remainder ? 1 : 0);

    blockIdx = args[t].endBlock;

    pthread_create(&threads[t], NULL, processBlock, &args[t]);
  }

  for (int t = 0; t < numThreads; t++)
    pthread_join(threads[t], NULL);
}

void applyFilterParallelByBlocks32(const IplImage *src, IplImage *dst,
                                   const Filter *f) {
  applyFilterParallelByBlocks(src, dst, f, 32, 32);
}

void applyFilterParallelByBlocks64(const IplImage *src, IplImage *dst,
                                   const Filter *f) {
  applyFilterParallelByBlocks(src, dst, f, 64, 64);
}

void applyFilterParallelByBlocks128(const IplImage *src, IplImage *dst,
                                    const Filter *f) {
  applyFilterParallelByBlocks(src, dst, f, 128, 128);
}

const NamedStrategy allStrategies[NUM_STRATEGIES] = {
    {"sequential", applyFilter},
    {"pixelwise", applyFilterParallelPixelwise},
    {"rows", applyFilterParallelByRows},
    {"cols", applyFilterParallelByCols},
    {"blocks32", applyFilterParallelByBlocks32},
    {"blocks64", applyFilterParallelByBlocks64},
    {"blocks128", applyFilterParallelByBlocks128},
};