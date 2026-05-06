# ─────────────────────────────────────────────
#  image_processing — Makefile
#  Requires: nvcc, g++, OpenCV 4.x, CUDA toolkit
# ─────────────────────────────────────────────

# ── Compiler & flags ────────────────────────
CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -Wall
NVCCFLAGS := -std=c++17 -O2

# ── OpenCV ──────────────────────────────────
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4)

# ── CUDA ────────────────────────────────────
CUDA_LIBS := -lcudart

# ── Directories ─────────────────────────────
SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build

# ── Sources & objects ───────────────────────
CPP_SRCS := $(SRC_DIR)/main.cpp \
             $(SRC_DIR)/cpu_filters.cpp \
             $(SRC_DIR)/utils.cpp \
             $(SRC_DIR)/benchmark.cpp

CU_SRCS  := $(SRC_DIR)/gpu_filters.cu

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,  $(BUILD_DIR)/%.o, $(CU_SRCS))

TARGET := image_processing

# ─────────────────────────────────────────────
.PHONY: all clean

all: $(BUILD_DIR) $(TARGET)

# ── Link everything with nvcc ────────────────
# nvcc must do the final link when CUDA objects are involved
$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(OPENCV_LIBS) $(CUDA_LIBS)

# ── Compile C++ sources ──────────────────────
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) $(OPENCV_CFLAGS) -c $< -o $@

# ── Compile CUDA sources ─────────────────────
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) $(OPENCV_CFLAGS) -c $< -o $@

# ── Create build dir ─────────────────────────
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Clean ────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) $(TARGET)