#ifndef TANH_H
#define TANH_H

#include "utils.h"

class TanhOp : public Object
{
 public:
  TanhOp(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out)
  {
    mat_ = in;
    out_ = std::shared_ptr<MatWdw>(new MatWdw(mat_->size_[0], mat_->size_[1]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Tanh(mat_->w_, out_->w_);

    return out_;
  }

  void Backward()
  {
    math->TanhDeriv(out_->dw_, out_->w_, mat_->dw_);
  }

  void ClearDw()
  {
    std::fill(mat_->dw_->data_.begin(), mat_->dw_->data_.end(), 0);
  }

  std::shared_ptr<MatWdw> mat_;
  std::shared_ptr<MatWdw> out_;
};

#endif
