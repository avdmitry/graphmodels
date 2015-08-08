#ifndef MUL_H
#define MUL_H

#include "utils.h"

// Multiply matrices.
class MulOp : public Object
{
 public:
  MulOp(std::shared_ptr<MatWdw> &mat1, std::shared_ptr<MatWdw> &mat2,
        std::shared_ptr<MatWdw> *out)
  {
    assert(mat1->size_[1] == mat2->size_[0]);
    mat1_ = mat1;
    mat2_ = mat2;
    out_ = std::shared_ptr<MatWdw>(new MatWdw(mat1_->size_[0], mat2_->size_[1]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Mul(mat1_->w_, mat2_->w_, out_->w_);

    return out_;
  }

  void Backward()
  {
    math->MulDeriv(mat1_->w_, mat2_->w_, mat1_->dw_,
                   mat2_->dw_, out_->dw_);
  }

  void ClearDw()
  {
    std::fill(mat1_->dw_->data_.begin(), mat1_->dw_->data_.end(), 0);
    std::fill(mat2_->dw_->data_.begin(), mat2_->dw_->data_.end(), 0);
  }

  std::shared_ptr<MatWdw> mat1_, mat2_;
  std::shared_ptr<MatWdw> out_;
};

#endif
