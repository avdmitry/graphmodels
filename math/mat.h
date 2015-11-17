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

  Mat(const std::vector<int>& sizes, bool with_dw = true)
      : data_device_(nullptr)
  {
    std::vector<int> sizes_tmp(sizes);
    while (sizes_tmp.size() < 4)
    {
      sizes_tmp.emplace_back(1);
    }

    int total = 1;
    for (int curr : sizes_tmp)
    {
      total *= curr;
      size_.emplace_back(curr);
    }
    data_.resize(total, 0);

    if (with_dw)
    {
      dw_ = std::shared_ptr<Mat>(new Mat(size_, false));
    }
  }

  Mat(int n, int d, int m = 1, int f = 1, bool with_dw = true)
      : data_device_(nullptr)
  {
    data_.resize(n * d * m * f, 0);

    size_.emplace_back(n);
    size_.emplace_back(d);
    size_.emplace_back(m);
    size_.emplace_back(f);

    if (with_dw)
    {
      dw_ = std::shared_ptr<Mat>(new Mat(size_, false));
    }
  }

  std::vector<int> size_;
  std::vector<float> data_;

  std::shared_ptr<Mat> dw_;
  float* data_device_;
};

#endif
