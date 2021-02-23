cd cmake-build-debug/
cmake -DCMAKE_PREFIX_PATH="$PWD"/../packages/libtorch -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" ../
