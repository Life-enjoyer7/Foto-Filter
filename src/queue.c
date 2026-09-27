#include "queue.h"
#include <stdio.h>
#include <stdlib.h>

Queue *queue_create(int capacity) {
  Queue *q = malloc(sizeof(Queue));
  if (!q)
    return NULL;

  q->buffer = malloc(sizeof(void *) * capacity);
  if (!q->buffer) {
    free(q);
    return NULL;
  }

  q->capacity = capacity;
  q->head = 0;
  q->tail = 0;
  q->count = 0;

  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);

  return q;
}

void queue_destroy(Queue *q) {
  if (!q)
    return;

  free(q->buffer);
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
  free(q);
}

void queue_push(Queue *q, void *item) {
  pthread_mutex_lock(&q->mutex);

  while (q->count == q->capacity) {
    pthread_cond_wait(&q->not_full, &q->mutex);
  }

  q->buffer[q->tail] = item;
  q->tail = (q->tail + 1) % q->capacity;
  q->count++;

  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
}

void *queue_pop(Queue *q) {
  pthread_mutex_lock(&q->mutex);

  while (q->count == 0) {
    pthread_cond_wait(&q->not_empty, &q->mutex);
  }

  void *item = q->buffer[q->head];
  q->head = (q->head + 1) % q->capacity;
  q->count--;

  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mutex);

  return item;
}

int queue_count(Queue *q) {
  pthread_mutex_lock(&q->mutex);
  int count = q->count;
  pthread_mutex_unlock(&q->mutex);
  return count;
}
