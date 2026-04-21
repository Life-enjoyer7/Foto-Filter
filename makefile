CC = g++
CFLAGS = -Wall -g -O2
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
BUILD_DIR = build
NAME = filter

OPENCV_LIB_PATH = /usr/local/opencv3.4/lib

run: $(BUILD_DIR)/$(NAME)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME)

$(BUILD_DIR)/$(NAME): $(MAIN_PATH) $(FILTER_PATH) | $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS) $(GTKFLAGS) $(PTHREADFLAGS) $(MATHFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@