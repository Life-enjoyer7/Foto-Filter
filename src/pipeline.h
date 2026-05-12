#ifndef PIPELINE_H
#define PIPELINE_H

#include "queue.h"
#include "filter.h"
#include "job.h"

typedef struct
{
    Queue *input_queue;
    const char **input_paths;
    const char **output_paths;
    int num_images;
    int filter_id;
    int strategy_id;
    int num_workers;
} ReaderArgs;

typedef struct
{
    Queue *input_queue;
    Queue *output_queue;
    Filter *filters;
    int strategy_id;
} WorkerArgs;

typedef struct
{
    Queue *output_queue;
    int num_images;
    int num_workers; // ← добавить
} WriterArgs;

void pipeline_run(const char **input_paths, const char **output_paths,
                  int num_images, int filter_id, int strategy_id,
                  int num_workers);

#endif