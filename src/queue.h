#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>

typedef struct {
  void **buffer;
  int capacity;
  int head;
  int tail;
  int count;
  pthread_mutex_t mutex;
  pthread_cond_t not_empty;
  pthread_cond_t not_full;
} Queue;

Queue *queue_create(int capacity);
void queue_destroy(Queue *q);
void queue_push(Queue *q, void *item);
void *queue_pop(Queue *q);
int queue_count(Queue *q);

#endif
