CC = g++
CFLAGS = -Wall -g -O2
INCLUDES = -I src -I/usr/local/opencv3.4/include

OPENCV_LIBS = -L/usr/local/opencv3.4/lib \
              -lopencv_highgui \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -Wl,-rpath,/usr/local/opencv3.4/lib

PTHREADFLAGS = -pthread

# Исходные файлы
MAIN_PATH = src/main.c
FILTER_PATH = src/filter.c
PIPELINE_PATH = src/pipeline.c
UTILS_PATH = src/utils.c
QUEUE_PATH = src/queue.c
JOB_PATH = src/job.c

# Тестовые файлы
TEST_PATH = tests/tests.c

BUILD_DIR = build
NAME = filter
NAME_TEST = tests

OPENCV_LIB_PATH = /usr/local/opencv3.4/lib

# Основная программа (конвейерная обработка)
build: $(BUILD_DIR)/$(NAME)

# Тесты (используют pipeline, queue, job, utils для тестирования конвейера)
test: $(BUILD_DIR)/$(NAME_TEST)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME_TEST)

# Сборка основной программы со всеми модулями
$(BUILD_DIR)/$(NAME): $(MAIN_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH) \
	      $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

# Сборка тестов (теперь тоже используют pipeline, queue, job, utils)
$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH) \
	      $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

# Очистка папки new_images (удаляет содержимое, но оставляет саму папку)
clean_images:
	rm -rf new_images/*

# Полная очистка (удаляет build и содержимое new_images)
clean_all: clean clean_images

clean:
	rm -rf $(BUILD_DIR)

.PHONY: build test clean clean_images clean_all