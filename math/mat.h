#ifndef MATH_MAT_H
#define MATH_MAT_H

#include <vector>
#include <memory>

class Mat
{
 public:
  Mat() : data_device_(nullptr)
  {
  }
  ~Mat();

  Mat(std::vector<int> sizes) : data_device_(nullptr)
  {
    int total = 1;
    for (int i = 0; i < sizes.size(); ++i)
    {
      int curr = sizes[i];
      total *= curr;
      size_.emplace_back(curr);
    }
    data_.resize(total, 0);
  }

  Mat(int n, int d = 1, int m = 1, int f = 1) : data_device_(nullptr)
  {
    data_.resize(n * d * m * f, 0);

    size_.emplace_back(n);
    size_.emplace_back(d);
    size_.emplace_back(m);
    size_.emplace_back(f);
  }

  std::vector<int> size_;
  std::vector<float> data_;

  float* data_device_;
};

#endif
