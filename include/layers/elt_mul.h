#ifndef EltMul_H
#define EltMul_H

#include "utils.h"

class EltMulOp : public Object
{
 public:
  EltMulOp(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
           std::shared_ptr<Mat> *out)
  {
    assert(mat1->w_.size() == mat2->w_.size());
    mat1_ = mat1;
    mat2_ = mat2;
    out_ = std::shared_ptr<Mat>(new Mat(mat1_->n_, mat1_->d_));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    for (int i = 0; i < mat1_->w_.size(); i++)
    {
      out_->w_[i] = mat1_->w_[i] * mat2_->w_[i];
    }

    return out_;
  }

  void Backward()
  {
    for (int i = 0; i < mat1_->w_.size(); i++)
    {
      mat1_->dw_[i] += mat2_->w_[i] * out_->dw_[i];
      mat2_->dw_[i] += mat1_->w_[i] * out_->dw_[i];
    }
  }

  void ClearDw() {
      std::fill(mat1_->dw_.begin(), mat1_->dw_.end(), 0);
      std::fill(mat2_->dw_.begin(), mat2_->dw_.end(), 0);
  }

  std::shared_ptr<Mat> mat1_, mat2_;
  std::shared_ptr<Mat> out_;
};

#endif
