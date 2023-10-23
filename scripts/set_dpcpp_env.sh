#!/bin/bash

GPU_VENDOR=intel


if [ "x$1" != "x" ]; then
    GPU_VENDOR=$1
fi

# check parameter
if [ "$GPU_VENDOR" != "intel" ] && [ "$GPU_VENDOR" != "nvidia" ]; then
    echo "Invalid gpu vendor: $GPU_VENDOR. Please choose from [intel, nvidia]."
    exit 1
fi

# set dpcpp system variables
source ~/intel/oneapi/setvars.sh #--include-intel-llvm

# build dpcpp toolchain 
if [ "x$DPCPP_HOME" == "x" ]; then
    DPCPP_HOME=~
fi

if [ ! -d "$DPCPP_HOME/llvm" ]; then
    git clone --config core.autocrlf=false https://github.com/intel/llvm $DPCPP_HOME/llvm -b sycl
    export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
    export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH

    if [ "$GPU_VENDOR" == "intel" ]; then
        python $DPCPP_HOME/llvm/buildbot/configure.py
        python $DPCPP_HOME/llvm/buildbot/compile.py
    else
        python $DPCPP_HOME/llvm/buildbot/configure.py --cuda 
        python $DPCPP_HOME/llvm/buildbot/compile.py
    fi
fi

# compile
if [ "$GPU_VENDOR" == "nvidia" ]; then
    cd extensions/dpcpp && make
    cd ../..
else
    export CPLUS_INCLUDE_PATH=/usr/include/c++/12
    cd extensions/dpcpp && make DEVICE=INTEL_GPU
    cd ../..
fi
