#ifndef PIPELINE_H
#define PIPELINE_H

#include "filter.h"
#include "job.h"
#include "queue.h"

typedef struct {
  Queue *input_queue;
  const char *const *input_paths;
  const char *const *output_paths;
  int num_images;
  int filter_id;
  int strategy_id;
  int num_workers;
} ReaderArgs;

typedef struct {
  Queue *input_queue;
  Queue *output_queue;
  Filter *filters;
  int strategy_id;
} WorkerArgs;

typedef struct {
  Queue *output_queue;
  int num_workers;
} WriterArgs;

void pipeline_run(const char *const *input_paths,
                  const char *const *output_paths, int num_images,
                  int filter_id, int strategy_id, int num_workers,
                  int queue_capacity);

#endif
