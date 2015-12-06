#include "mat.h"

#include "common.h"

Mat::Mat() : data_device_(nullptr)
{
}

Mat::Mat(const std::vector<int>& sizes, bool with_dw)
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

Mat::Mat(int n, int d, int m, int f, bool with_dw)
    : data_device_(nullptr)
{
  size_.emplace_back(n);
  size_.emplace_back(d);
  size_.emplace_back(m);
  size_.emplace_back(f);
  data_.resize(n * d * m * f, 0);

  if (with_dw)
  {
    dw_ = std::shared_ptr<Mat>(new Mat(size_, false));
  }
}

Mat::~Mat()
{
  if (data_device_ != nullptr)
  {
    math->MemoryFree(data_device_);
    data_device_ = nullptr;
  }
}
