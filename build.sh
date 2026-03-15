set -e 
rm -rf build
mkdir build
cmake -B ./build # 效果等同于cd bulid && cmake ..
cmake --build ./build
