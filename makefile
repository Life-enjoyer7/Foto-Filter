CC = gcc
CFLAGS = -std=gnu11 -Wall -g -O2

# Путь к OpenCV 3.x (нужен C API); в CI переопределяется: make OPENCV_PREFIX=...
OPENCV_PREFIX ?= /usr/local/opencv3.4
INCLUDES = -I src -I$(OPENCV_PREFIX)/include

OPENCV_LIBS = -L$(OPENCV_PREFIX)/lib \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -lstdc++ -lm \
              -Wl,-rpath,$(OPENCV_PREFIX)/lib

PTHREADFLAGS = -pthread

MAIN_PATH = src/main.c src/main_utils.c
FILTER_PATH = src/filter.c
TEST_PATH = tests/tests.c tests/tests_utils.c src/main_utils.c
BENCH_PATH = benchmarks/bench.c src/main_utils.c tests/tests_utils.c
BUILD_DIR = build
NAME = filter
NAME_TEST = tests
NAME_BENCH = bench

REPEAT ?= 5
BENCH_OUT_DIR = benchmarks/generated

build: $(BUILD_DIR)/$(NAME)

test: $(BUILD_DIR)/$(NAME_TEST)
	$(BUILD_DIR)/$(NAME_TEST)

bench-bin: $(BUILD_DIR)/$(NAME_BENCH)

benchmark: bench-bin
	mkdir -p $(BENCH_OUT_DIR)
	$(BUILD_DIR)/$(NAME_BENCH) $(REPEAT) > $(BENCH_OUT_DIR)/bench_results.csv
	python3 benchmarks/plot.py $(BENCH_OUT_DIR)/bench_results.csv $(BENCH_OUT_DIR)

clean_benchmark:
	rm -rf $(BENCH_OUT_DIR)

$(BUILD_DIR)/$(NAME): $(MAIN_PATH) src/main_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) tests/tests_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

$(BUILD_DIR)/$(NAME_BENCH): $(BENCH_PATH) src/main_utils.h tests/tests_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(BENCH_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: build test bench-bin benchmark clean clean_benchmark
