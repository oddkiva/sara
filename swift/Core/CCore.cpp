#include "CCore.hpp"

#include <DO/Sara/Core/TicToc.hpp>

void tic() {
  DO::Sara::tic();
}

void toc(const char *msg) {
  DO::Sara::toc(msg);
}

void square(int* numbers, int n)
{
  for (int i = 0; i < n; ++i)
    numbers[i] *= numbers[i];
}
