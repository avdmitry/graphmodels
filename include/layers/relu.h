#ifndef Relu_H
#define Relu_H

#include "utils.h"

class ReluOp : public Object
{
 public:
  ReluOp(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out)
  {
    mat_ = in;
    out_ = std::shared_ptr<Mat>(new Mat(mat_->n_, mat_->d_));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    for (int i = 0; i < mat_->w_.size(); i++)
    {
      out_->w_[i] = std::max(0.0f, mat_->w_[i]);
    }

    return out_;
  }

  void Backward()
  {
    for (size_t i = 0; i < mat_->w_.size(); i++)
    {
      if (mat_->w_[i] > 0)
      {
        mat_->dw_[i] += out_->dw_[i];
      }
    }
  }

  void ClearDw() {
      std::fill(mat_->dw_.begin(), mat_->dw_.end(), 0);
  }

  std::shared_ptr<Mat> mat_;
  std::shared_ptr<Mat> out_;
};

#endif
