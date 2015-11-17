#ifndef RELU_H
#define RELU_H

#include "utils.h"

class ReluOp : public Object
{
 public:
  ReluOp(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out)
  {
    in_ = in;
    out_ = std::shared_ptr<Mat>(new Mat(in_->size_));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    math->Relu(in_, out_);

    return out_;
  }

  void Backward()
  {
    math->ReluDeriv(in_, in_->dw_, out_, out_->dw_);
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
