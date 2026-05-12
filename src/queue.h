#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>

typedef struct
{
    void **buffer;            // указатель на массив указателей
    int capacity;             // максимальное количество элементов
    int head;                 // индекс для чтения (откуда забирать)
    int tail;                 // индекс для записи (куда класть)
    int count;                // текущее количество элементов
    pthread_mutex_t mutex;    // мьютекс для синхронизации
    pthread_cond_t not_empty; // условная переменная: "очередь не пуста"
    pthread_cond_t not_full;  // условная переменная: "очередь не полна"
} Queue;

Queue *queue_create(int capacity);
void queue_destroy(Queue *q);
void queue_push(Queue *q, void *item);
void *queue_pop(Queue *q);
int queue_count(Queue *q);

#endif