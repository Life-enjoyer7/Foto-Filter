// job.h
#ifndef JOB_H
#define JOB_H

#include <opencv2/core/core_c.h>

typedef struct
{
    int id;
    char *input_path;
    char *output_path;
    IplImage *image;
    IplImage *result;
    int filter_id;
    int strategy_id;
} Job;

Job *job_create(int id, const char *input_path, const char *output_path);
void job_destroy(Job *job);

#endif