#ifndef TanhRow_H
#define TanhRow_H

#include "utils.h"

class TanhOp : public Object
{
 public:
  TanhOp(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out)
  {
    mat_ = in;
    out_ = std::shared_ptr<Mat>(new Mat(mat_->n_, mat_->d_));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    for (int i = 0; i < mat_->w_.size(); i++)
    {
      out_->w_[i] = tanh(mat_->w_[i]);
    }

    return out_;
  }

  void Backward()
  {
    for (size_t i = 0; i < mat_->w_.size(); i++)
    {
      float mwi = out_->w_[i];
      mat_->dw_[i] += (1.0 - mwi * mwi) * out_->dw_[i];
    }
  }

  void ClearDw() {
      std::fill(mat_->dw_.begin(), mat_->dw_.end(), 0);
  }

  std::shared_ptr<Mat> mat_;
  std::shared_ptr<Mat> out_;
};

#endif
