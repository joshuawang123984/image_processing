CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build

CPP_SRCS := $(SRC_DIR)/main.cpp \
             $(SRC_DIR)/cpu_filters.cpp \
             $(SRC_DIR)/utils.cpp

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SRCS))

TARGET := image_processing

.PHONY: all clean

all: $(BUILD_DIR) $(TARGET)

$(TARGET): $(CPP_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)