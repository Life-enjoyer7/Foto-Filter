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
TEST_PATH = tests/tests.c tests/utils_tests.c

# Бенчмарк
BENCH_PATH = benchmarks/bench.c tests/utils_tests.c

BUILD_DIR = build
NAME = filter
NAME_TEST = tests
NAME_BENCH = bench

# Число повторов на каждую точку замера в `make benchmark` (без учёта прогрева).
REPEAT ?= 2
BENCH_OUT_DIR = benchmarks/generated

# Основная программа (конвейерная обработка)
build: $(BUILD_DIR)/$(NAME)

# Тесты
test: $(BUILD_DIR)/$(NAME_TEST)
	$(BUILD_DIR)/$(NAME_TEST)

bench-bin: $(BUILD_DIR)/$(NAME_BENCH)

# Полный перебор фильтр x стратегия x число воркеров -> CSV -> PNG-графики.
benchmark: bench-bin
	mkdir -p $(BENCH_OUT_DIR) $(BENCH_OUT_DIR)/bench_out
	$(BUILD_DIR)/$(NAME_BENCH) $(REPEAT) | grep -E "^filter,strategy,workers,num_images|blur3x3|blur5x5|gaussian3x3|gaussian5x5|motionblur|findedges1|findedges2|findedges3|findedges4|sharpen1|sharpen2|sharpen3|emboss1|emboss2|identity" > $(BENCH_OUT_DIR)/bench_results.csv
	python3 benchmarks/plot.py $(BENCH_OUT_DIR)/bench_results.csv $(BENCH_OUT_DIR)
	rm -rf $(BENCH_OUT_DIR)/bench_out

# Сборка основной программы со всеми модулями
$(BUILD_DIR)/$(NAME): $(MAIN_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH) \
	      $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

# Сборка тестов
$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) tests/utils_tests.h $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH) \
	      $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

# Сборка бенчмарка
$(BUILD_DIR)/$(NAME_BENCH): $(BENCH_PATH) tests/utils_tests.h $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(BENCH_PATH) $(FILTER_PATH) $(PIPELINE_PATH) $(UTILS_PATH) $(QUEUE_PATH) $(JOB_PATH) \
	      $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@


clean_benchmark:
	rm -rf $(BENCH_OUT_DIR)



clean:
	rm -rf $(BUILD_DIR)

.PHONY: build test clean clean_images clean_all bench-bin benchmark