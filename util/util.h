#ifndef UTIL_H
#define UTIL_H

namespace util {

void init_data(float* input, size_t length);
void print_data(float* input, size_t high, size_t width);
void compare_result(float* res1, float* res2, size_t length);

}

#endif // UTIL_H