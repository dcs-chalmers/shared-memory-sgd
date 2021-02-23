#cmake -DCMAKE_PREFIX_PATH=/home/ivan/projects/mininn/packages/libtorch --build ./cmake-build-debug --target all -- -j 2
export OMP_DYNAMIC=FALSE
cmake --build ./cmake-build-debug --target all -- -j 2
