#include "job.h"
#include <opencv2/core/core_c.h>
#include <stdlib.h>
#include <string.h>

Job *job_create(int id, const char *input_path, const char *output_path) {
  Job *job = malloc(sizeof(Job));
  job->id = id;
  job->input_path = strdup(input_path);
  job->output_path = strdup(output_path);
  job->image = NULL;
  job->result = NULL;
  job->filter_id = 0;
  job->strategy_id = 0;
  return job;
}

void job_destroy(Job *job) {
  if (job) {
    free(job->input_path);
    free(job->output_path);
    if (job->image)
      cvReleaseImage(&job->image);
    if (job->result)
      cvReleaseImage(&job->result);
    free(job);
  }
}
