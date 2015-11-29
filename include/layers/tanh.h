#ifndef TANH_H
#define TANH_H

#include "utils.h"

class TanhOp : public Object
{
 public:
  TanhOp(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out)
  {
    in_ = in;
    out_ = std::shared_ptr<Mat>(new Mat(in_->size_));
    *out = out_;
  }

  void Forward(bool train)
  {
    math->Tanh(in_, out_);
  }

  void Backward()
  {
    math->TanhDeriv(in_, in_->dw_, out_, out_->dw_);
  }

  void SetBatchSize(int new_size)
  {
    in_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> out_;
};

#endif
