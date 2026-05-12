
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pipeline.h"

static void *reader_func(void *arg)
{
    ReaderArgs *args = (ReaderArgs *)arg;
    Queue *input_queue = args->input_queue;
    const char **input_paths = args->input_paths;
    const char **output_paths = args->output_paths;
    int num_images = args->num_images;
    int filter_id = args->filter_id;
    int strategy_id = args->strategy_id;
    int num_workers = args->num_workers;

    for (int i = 0; i < num_images; i++)
    {
        Job *job = job_create(i, input_paths[i], output_paths[i]);
        if (!job)
        {
            fprintf(stderr, "Reader: failed to create job for %s\n", input_paths[i]);
            continue;
        }

        job->filter_id = filter_id;
        job->strategy_id = strategy_id;

        job->image = cvLoadImage(job->input_path, CV_LOAD_IMAGE_COLOR);
        if (!job->image)
        {
            fprintf(stderr, "Reader: failed to load image: %s\n", job->input_path);
            job_destroy(job);
            continue;
        }

        printf("Reader: loaded %s (id=%d)\n", job->input_path, job->id);
        queue_push(input_queue, job);
    }

    for (int i = 0; i < num_workers; i++)
    {
        queue_push(input_queue, NULL);
    }

    printf("Reader: all %d images processed, sent %d poison pills\n",
           num_images, num_workers);

    return NULL;
}

static void *worker_func(void *arg)
{
    WorkerArgs *args = (WorkerArgs *)arg;
    Queue *input_queue = args->input_queue;
    Queue *output_queue = args->output_queue;
    Filter *filters = args->filters;
    int strategy_id = args->strategy_id;

    void (*apply_func)(const IplImage *, IplImage *, const Filter *);

    switch (strategy_id)
    {
    case 0:
        apply_func = applyFilterParallelPixelwise;
        break;
    case 1:
        apply_func = applyFilterParallelByRows;
        break;
    case 2:
        apply_func = applyFilterParallelByCols;
        break;
    case 3:
        apply_func = applyFilterParallelByBlocks32;
        break;
    case 4:
        apply_func = applyFilterParallelByBlocks64;
        break;
    case 5:
        apply_func = applyFilterParallelByBlocks128;
        break;
    default:
        apply_func = applyFilterParallelByRows;
        break;
    }

    while (1)
    {
        Job *job = (Job *)queue_pop(input_queue);

        if (job == NULL)
        {
            queue_push(output_queue, NULL); // Пробрасываем poison pill дальше
            break;
        }

        IplImage *result = cvCreateImage(cvGetSize(job->image),
                                         job->image->depth,
                                         job->image->nChannels);

        apply_func(job->image, result, &filters[job->filter_id]);

        job->result = result;

        queue_push(output_queue, job);
    }

    return NULL;
}

static void *writer_func(void *arg)
{
    WriterArgs *args = (WriterArgs *)arg;
    Queue *output_queue = args->output_queue;
    int num_images = args->num_images;
    int completed = 0;

    printf("Writer: ожидается %d изображений\n", num_images);

    while (completed < num_images)
    {
        Job *job = (Job *)queue_pop(output_queue);

        // Пропускаем poison pills - они не содержат изображения
        if (job == NULL)
        {
            printf("Writer: получен poison pill\n");
            continue;
        }

        if (!cvSaveImage(job->output_path, job->result))
        {
            fprintf(stderr, "Writer: failed to save %s\n", job->output_path);
        }
        else
        {
            printf("Writer: saved %s\n", job->output_path);
        }

        job_destroy(job);
        completed++;
    }

    printf("Writer: завершил работу, сохранено %d изображений\n", completed);
    return NULL;
}

void pipeline_run(const char **input_paths, const char **output_paths,
                  int num_images, int filter_id, int strategy_id,
                  int num_workers)
{
    printf("Pipeline: запуск с %d воркерами, фильтр %d, стратегия %d\n",
           num_workers, filter_id, strategy_id);

    Queue *input_queue = queue_create(10);
    Queue *output_queue = queue_create(10);

    Filter filters[15];
    filters[0] = filter_blur3x3();
    filters[1] = filter_blur5x5();
    filters[2] = filter_gaussian3x3();
    filters[3] = filter_gaussian5x5();
    filters[4] = filter_motionblur();
    filters[5] = filter_findedges1();
    filters[6] = filter_findedges2();
    filters[7] = filter_findedges3();
    filters[8] = filter_findedges4();
    filters[9] = filter_sharpen1();
    filters[10] = filter_sharpen2();
    filters[11] = filter_sharpen3();
    filters[12] = filter_emboss1();
    filters[13] = filter_emboss2();
    filters[14] = filter_identity();

    ReaderArgs reader_args = {
        .input_queue = input_queue,
        .input_paths = input_paths,
        .output_paths = output_paths,
        .num_images = num_images,
        .filter_id = filter_id,
        .strategy_id = strategy_id,
        .num_workers = num_workers};

    WorkerArgs worker_args = {
        .input_queue = input_queue,
        .output_queue = output_queue,
        .filters = filters,
        .strategy_id = strategy_id};

    WriterArgs writer_args = {
        .output_queue = output_queue,
        .num_images = num_images};

    pthread_t reader, writer;
    pthread_t workers[num_workers];

    pthread_create(&reader, NULL, reader_func, &reader_args);

    for (int i = 0; i < num_workers; i++)
    {
        pthread_create(&workers[i], NULL, worker_func, &worker_args);
    }

    pthread_create(&writer, NULL, writer_func, &writer_args);

    pthread_join(reader, NULL);
    printf("Pipeline: читатель завершил работу\n");

    for (int i = 0; i < num_workers; i++)
    {
        pthread_join(workers[i], NULL);
        printf("Pipeline: воркер %d завершил работу\n", i);
    }

    pthread_join(writer, NULL);
    printf("Pipeline: писатель завершил работу\n");

    for (int i = 0; i < 15; i++)
    {
        filter_free(&filters[i]);
    }

    queue_destroy(input_queue);
    queue_destroy(output_queue);

    printf("Pipeline: завершён\n");
}