#include "job.h"
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/core_c.h>

Job *job_create(int id, const char *input_path, const char *output_path)
{
    Job *job = (Job *)malloc(sizeof(Job));
    job->id = id;
    job->input_path = strdup(input_path); // strdup() - выделяет память под копию строки и возвращает указатель на неё.
    job->output_path = strdup(output_path);
    job->image = NULL;
    job->result = NULL;
    job->filter_id = 0;   // по умолчанию, например, blur3x3
    job->strategy_id = 0; // по умолчанию по пикселям
    return job;
}

void job_destroy(Job *job)
{
    if (job)
    {
        free(job->input_path);
        free(job->output_path);
        if (job->image)
            cvReleaseImage(&job->image);
        if (job->result)
            cvReleaseImage(&job->result);
        free(job);
    }
}