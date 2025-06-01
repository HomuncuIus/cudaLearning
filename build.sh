set -e 
cd build
rm -rf *
cmake ..
cmake --build .
