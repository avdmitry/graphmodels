#ifndef RELU_H
#define RELU_H

#include "utils.h"

class ReluOp : public Object
{
 public:
  ReluOp(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out)
  {
    mat_ = in;
    out_ = std::shared_ptr<MatWdw>(new MatWdw(mat_->size_[0], mat_->size_[1]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Relu(mat_->w_, out_->w_);

    return out_;
  }

  void Backward()
  {
    math->ReluDeriv(out_->dw_, out_->w_, mat_->dw_);
  }

  void ClearDw()
  {
    std::fill(mat_->dw_->data_.begin(), mat_->dw_->data_.end(), 0);
  }

  std::shared_ptr<MatWdw> mat_;
  std::shared_ptr<MatWdw> out_;
};

#endif
