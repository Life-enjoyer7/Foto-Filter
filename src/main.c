#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gtk/gtk.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include "filter.h"

// Определим константы, если их нет
#ifndef CV_LOAD_IMAGE_COLOR
#define CV_LOAD_IMAGE_COLOR 1
#endif

#define NUM_FILTERS 15

int main(int argc, char *argv[])
{
    char pass[100] = "";
    int filter;

    printf("Инициализация GTK...\n");
    gtk_init(&argc, &argv);
    printf("GTK инициализирован\n");

    Filter filters[NUM_FILTERS];
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

    char *filename_open = NULL;

    while (strcmp(pass, "exit") != 0)
    {
        GtkWidget *dialog_open = gtk_file_chooser_dialog_new(
            "Open File",
            NULL,
            GTK_FILE_CHOOSER_ACTION_OPEN,
            "_Cancel", GTK_RESPONSE_CANCEL,
            "_Open", GTK_RESPONSE_ACCEPT,
            NULL);

        gint res_open = gtk_dialog_run(GTK_DIALOG(dialog_open));

        if (res_open == GTK_RESPONSE_ACCEPT)
        {
            GtkFileChooser *chooser_open = GTK_FILE_CHOOSER(dialog_open);
            filename_open = gtk_file_chooser_get_filename(chooser_open);
            gtk_widget_destroy(dialog_open);

            while (gtk_events_pending())
                gtk_main_iteration();

            printf("Загружаю файл: %s\n", filename_open);
            IplImage *image = cvLoadImage(filename_open, CV_LOAD_IMAGE_COLOR);
            printf("image = %p\n", (void *)image);
            if (!image)
            {
                printf("Ошибка загрузки!\n");
                g_free(filename_open);
                continue;
            }

            IplImage *result = cvCreateImage(
                cvSize(image->width, image->height),
                IPL_DEPTH_8U,
                3);

            printf("Выберите фильтр (0-14):\n");
            printf("0-blur3x3, 1-blur5x5, 2-gaussian3x3, 3-gaussian5x5, 4-motionblur\n");
            printf("5-findedges1, 6-findedges2, 7-findedges3, 8-findedges4\n");
            printf("9-sharpen1, 10-sharpen2, 11-sharpen3, 12-emboss1, 13-emboss2, 14-identity\n");
            scanf("%d", &filter);
            while (getchar() != '\n')
                ; // очистка буфера

            applyFilter(image, result, &filters[filter]);

            GtkWidget *dialog_close = gtk_file_chooser_dialog_new(
                "Save File",
                NULL,
                GTK_FILE_CHOOSER_ACTION_SAVE,
                "_Cancel", GTK_RESPONSE_CANCEL,
                "_Save", GTK_RESPONSE_ACCEPT,
                NULL);

            GtkFileChooser *chooser_close = GTK_FILE_CHOOSER(dialog_close);
            gtk_file_chooser_set_do_overwrite_confirmation(chooser_close, TRUE);
            gtk_file_chooser_set_current_name(chooser_close, "Untitled document");

            gint res_close = gtk_dialog_run(GTK_DIALOG(dialog_close));
            if (res_close == GTK_RESPONSE_ACCEPT)
            {
                char *save_filename = gtk_file_chooser_get_filename(chooser_close);
                cvSaveImage(save_filename, result);
                printf("Сохранено: %s\n", save_filename);
                g_free(save_filename);
            }

            gtk_widget_destroy(dialog_close);

            while (gtk_events_pending())
                gtk_main_iteration();

            cvReleaseImage(&image);
            cvReleaseImage(&result);
            g_free(filename_open);
        }
        else
        {
            gtk_widget_destroy(dialog_open);
        }

        printf("Чтобы продолжить, введите любой символ. Для выхода нажмите exit\n");
        scanf("%s", pass);
        while (getchar() != '\n')
            ;
    }

    // Очистка фильтров
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        filter_free(&filters[i]);
    }

    return 0;
}