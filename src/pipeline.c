#include "pipeline.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PIPELINE_QUEUE_CAPACITY 10

static void *reader_func(void *arg) {
  ReaderArgs *args = (ReaderArgs *)arg;
  Queue *input_queue = args->input_queue;
  const char *const *input_paths = args->input_paths;
  const char *const *output_paths = args->output_paths;
  int num_images = args->num_images;
  int filter_id = args->filter_id;
  int strategy_id = args->strategy_id;
  int num_workers = args->num_workers;

  for (int i = 0; i < num_images; i++) {
    Job *job = job_create(i, input_paths[i], output_paths[i]);
    if (!job) {
      fprintf(stderr, "Reader: failed to create job for %s\n", input_paths[i]);
      continue;
    }

    job->filter_id = filter_id;
    job->strategy_id = strategy_id;

    job->image = cvLoadImage(job->input_path, CV_LOAD_IMAGE_COLOR);
    if (!job->image) {
      fprintf(stderr, "Reader: failed to load image: %s\n", job->input_path);
      job_destroy(job);
      continue;
    }

    printf("Reader: loaded %s (id=%d)\n", job->input_path, job->id);
    queue_push(input_queue, job);
  }

  for (int i = 0; i < num_workers; i++) {
    queue_push(input_queue, NULL);
  }

  printf("Reader: sent %d poison pills\n", num_workers);
  return NULL;
}

static void *worker_func(void *arg) {
  WorkerArgs *args = (WorkerArgs *)arg;
  Queue *input_queue = args->input_queue;
  Queue *output_queue = args->output_queue;
  Filter *filters = args->filters;
  int strategy_id = args->strategy_id;

  FilterFn apply_func = allStrategies[strategy_id].fn;

  while (1) {
    Job *job = queue_pop(input_queue);

    if (job == NULL) {
      queue_push(output_queue, NULL);
      break;
    }

    IplImage *result = cvCreateImage(cvGetSize(job->image), job->image->depth,
                                     job->image->nChannels);

    apply_func(job->image, result, &filters[job->filter_id]);

    job->result = result;

    queue_push(output_queue, job);
  }

  return NULL;
}

static void *writer_func(void *arg) {
  WriterArgs *args = (WriterArgs *)arg;
  Queue *output_queue = args->output_queue;
  int num_workers = args->num_workers;
  int completed = 0;
  int poisons = 0;

  while (poisons < num_workers) {
    Job *job = queue_pop(output_queue);

    if (job == NULL) {
      poisons++;
      continue;
    }

    if (!cvSaveImage(job->output_path, job->result, 0)) {
      fprintf(stderr, "Writer: failed to save %s\n", job->output_path);
    } else {
      printf("Writer: saved %s\n", job->output_path);
    }

    job_destroy(job);
    completed++;
  }

  printf("Writer: saved %d images\n", completed);
  return NULL;
}

void pipeline_run(const char *const *input_paths,
                  const char *const *output_paths, int num_images,
                  int filter_id, int strategy_id, int num_workers) {
  printf("Pipeline: %d workers, filter %d, strategy %d\n", num_workers,
         filter_id, strategy_id);

  Queue *input_queue = queue_create(PIPELINE_QUEUE_CAPACITY);
  Queue *output_queue = queue_create(PIPELINE_QUEUE_CAPACITY);

  Filter filters[NUM_FILTERS];
  for (int i = 0; i < NUM_FILTERS; i++)
    filters[i] = filter_by_id(i);

  int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    cores = 4;
  int per_worker = cores / num_workers;
  if (per_worker < 1)
    per_worker = 1;
  set_conv_threads(per_worker);

  ReaderArgs reader_args = {.input_queue = input_queue,
                            .input_paths = input_paths,
                            .output_paths = output_paths,
                            .num_images = num_images,
                            .filter_id = filter_id,
                            .strategy_id = strategy_id,
                            .num_workers = num_workers};

  WorkerArgs worker_args = {.input_queue = input_queue,
                            .output_queue = output_queue,
                            .filters = filters,
                            .strategy_id = strategy_id};

  WriterArgs writer_args = {.output_queue = output_queue,
                            .num_workers = num_workers};

  pthread_t reader, writer;
  pthread_t workers[num_workers];

  pthread_create(&reader, NULL, reader_func, &reader_args);

  for (int i = 0; i < num_workers; i++) {
    pthread_create(&workers[i], NULL, worker_func, &worker_args);
  }

  pthread_create(&writer, NULL, writer_func, &writer_args);

  pthread_join(reader, NULL);
  printf("Pipeline: reader done\n");

  for (int i = 0; i < num_workers; i++) {
    pthread_join(workers[i], NULL);
    printf("Pipeline: worker %d done\n", i);
  }

  pthread_join(writer, NULL);
  printf("Pipeline: writer done\n");

  for (int i = 0; i < NUM_FILTERS; i++)
    filter_free(&filters[i]);

  queue_destroy(input_queue);
  queue_destroy(output_queue);
}
