#ifndef SIGM_H
#define SIGM_H

#include "utils.h"

class SigmOp : public Object
{
 public:
  SigmOp(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out)
  {
    mat_ = in;
    out_ = std::shared_ptr<MatWdw>(new MatWdw(mat_->size_[0], mat_->size_[1]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Sigm(mat_->w_, out_->w_);

    return out_;
  }

  void Backward()
  {
    math->SigmDeriv(out_->dw_, out_->w_, mat_->dw_);
  }

  void ClearDw()
  {
    std::fill(mat_->dw_->data_.begin(), mat_->dw_->data_.end(), 0);
  }

  std::shared_ptr<MatWdw> mat_;
  std::shared_ptr<MatWdw> out_;
};

#endif
