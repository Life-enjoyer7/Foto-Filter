CC = g++
CFLAGS = -Wall -g  -O3 -march=native
INCLUDES = -I src -I/usr/local/opencv3.4/include

OPENCV_LIBS = -L/usr/local/opencv3.4/lib \
              -lopencv_highgui \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -Wl,-rpath,/usr/local/opencv3.4/lib

GTKFLAGS = $(shell pkg-config --cflags --libs gtk+-3.0)
PTHREADFLAGS = -pthread
MATHFLAGS = -lm

MAIN_PATH = src/main.c
FILTER_PATH = src/filter.c
TEST_PATH = tests/tests.c
BUILD_DIR = build
NAME = filter
NAME_TEST = tests

OPENCV_LIB_PATH = /usr/local/opencv3.4/lib

run: $(BUILD_DIR)/$(NAME)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME)

test: $(BUILD_DIR)/$(NAME_TEST)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME_TEST)

$(BUILD_DIR)/$(NAME): $(MAIN_PATH) $(FILTER_PATH) | $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(GTKFLAGS) $(PTHREADFLAGS) $(MATHFLAGS) -o $@

$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) $(FILTER_PATH) | $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(PTHREADFLAGS) $(MATHFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

.PHONY: run test clean