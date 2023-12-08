ifndef DEVICE
    DEVICE=NVIDIA_GPU
endif

ifeq ($(DEVICE), NVIDIA_GPU)
    SYCLTARGETS=nvptx64-nvidia-cuda
    CXXFLAGS += -DNVIDIA_GPU=1
else ifeq ($(DEVICE), INTEL_GPU)
    SYCLTARGETS="spir64_gen"
    CXXFLAGS += -DINTEL_GPU=1
else
    SYCLTARGETS="spir64_x86_64"
endif

#CXX := icpx -fsycl -v -fsycl-targets=$(SYCLTARGETS)
CXX := clang++ -fsycl -fsycl-targets=$(SYCLTARGETS) -Xclang -fdenormal-fp-math=ieee

LD := ld
SOURCES := $(wildcard *.cpp)

ifeq ($(DEVICE), INTEL_GPU)
CXX += -Xs "-device dg2" -v
endif

# Output directory for .o files and dynamic library
BUILD_DIR = build

SOFLAGS = -shared

CONDA_ENV = $(CONDA_PREFIX)
PYTHON_VERSION = $(shell python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]))')

INC_PATH = -I$(CONDA_ENV)/include/python$(PYTHON_VERSION)
INC_PATH += -I$(CONDA_ENV)/lib/python$(PYTHON_VERSION)/site-packages/torch/include/torch/csrc/api/include
INC_PATH += -I$(CONDA_ENV)/lib/python$(PYTHON_VERSION)/site-packages/torch/include
INC_PATH += -I$(CONDA_ENV)/lib/python$(PYTHON_VERSION)/site-packages/torch/include/ATen
INC_PATH += -I$(CONDA_ENV)/lib/python$(PYTHON_VERSION)/site-packages/numpy/core/include

CXXFLAGS += $(INC_PATH)
CXXFLAGS += -fPIC -Ofast -std=c++17
#CXXFLAGS += -Wno-deprecated-declarations

LDFLAGS += -L$(CONDA_ENV)/lib -lpython$(PYTHON_VERSION)
#LDFLAGS += -L$(CONDA_ENV)/lib/python$(PYTHON_VERSION)/site-packages/torch/lib -lc10 -ltorch_global_deps -ltorch_cuda_linalg -lshm -ltorch_python -ltorch -ltorch_cuda -ltorch_cpu
LDFLAGS += -L$(CONDA_ENV)/lib/python$(PYTHON_VERSION)/site-packages/torch/lib -lc10 -ltorch_python

# Rule to create the build directory if it doesn't exist
$(shell mkdir -p $(BUILD_DIR))

SOURCES_DPCPP := bindings.cpp rays.cpp gridencoder.dp.cpp #segment_coo.dp.cpp

# List of object files for gridencoder module
OBJECTS_DPCPP := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SOURCES_DPCPP))

# Target dynamic library for gridencoder module
# DPCPP_LIB := _gridencoder_dpcpp.so
DPCPP_LIB := _dpcpp_backend.so
TARGET_LIBS = $(DPCPP_LIB)

# Rule for creating the build directory
$(shell mkdir -p $(BUILD_DIR))

# Rule for compiling .cpp files and generating .o files in the build directory
$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Default target
.PHONY: all
all: $(DPCPP_LIB)

$(DPCPP_LIB): $(OBJECTS_DPCPP)
	$(CXX) $(LDFLAGS) -shared $^ -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(DPCPP_LIB)

