CC = g++
CFLAGS = -Wall -g -O2
INCLUDES = -I src -I/usr/local/opencv3.4/include

OPENCV_LIBS_TEST = -L/usr/local/opencv3.4/lib \
              -lopencv_highgui \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \

OPENCV_LIBS_MAIN = -L/usr/local/opencv3.4/lib \
              -lopencv_highgui \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -Wl,-rpath,/usr/local/opencv3.4/lib

PTHREADFLAGS = -pthread


MAIN_PATH = src/main.c
FILTER_PATH = src/filter.c
TEST_PATH = tests/tests.c
BUILD_DIR = build
NAME = filter
NAME_TEST = tests

OPENCV_LIB_PATH = /usr/local/opencv3.4/lib

build: $(BUILD_DIR)/$(NAME)

test: $(BUILD_DIR)/$(NAME_TEST)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME_TEST)

$(BUILD_DIR)/$(NAME): $(MAIN_PATH) $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_MAIN) $(PTHREADFLAGS)  -o $@

$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_TEST) $(PTHREADFLAGS)  -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: build test clean