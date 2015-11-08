#ifndef RELU_H
#define RELU_H

#include "utils.h"

class ReluOp : public Object
{
 public:
  ReluOp(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out)
  {
    in_ = in;
    out_ = std::shared_ptr<MatWdw>(
        new MatWdw(in_->size_[0], in_->size_[1], in_->size_[2], in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Relu(in_->w_, out_->w_);

    return out_;
  }

  void Backward()
  {
    math->ReluDeriv(in_->dw_, out_->w_, out_->dw_);
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
  }

  void GetParams(std::vector<std::shared_ptr<MatWdw>> &params)
  {
  }

  std::shared_ptr<MatWdw> in_;
  std::shared_ptr<MatWdw> out_;
};

#endif
