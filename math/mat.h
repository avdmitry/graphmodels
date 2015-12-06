#ifndef MATH_MAT_H
#define MATH_MAT_H

#include <vector>
#include <memory>

class Mat
{
 public:
  Mat();
  Mat(const std::vector<int>& sizes, bool with_dw = true);
  Mat(int n, int d, int m = 1, int f = 1, bool with_dw = true);
  ~Mat();

  std::vector<int> size_;
  std::vector<float> data_;

  std::shared_ptr<Mat> dw_;
  float* data_device_;
};

#endif
