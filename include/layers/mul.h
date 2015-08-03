#ifndef Mul_H
#define Mul_H

#include "utils.h"

// Multiply matrices.
class MulOp : public Object
{
 public:
  MulOp(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
        std::shared_ptr<Mat> *out)
  {
    assert(mat1->d_ == mat2->n_);
    mat1_ = mat1;
    mat2_ = mat2;
    out_ = std::shared_ptr<Mat>(new Mat(mat1_->n_, mat2_->d_));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    for (int i = 0; i < mat1_->n_; i++)
    {  // loop over rows of m1
      for (int j = 0; j < mat2_->d_; j++)
      {  // loop over cols of m2
        float dot = 0.0;
        for (int k = 0; k < mat1_->d_; k++)
        {  // dot product loop
          dot += mat1_->w_[mat1_->d_ * i + k] * mat2_->w_[mat2_->d_ * k + j];
        }
        out_->w_[mat2_->d_ * i + j] = dot;
      }
    }

    return out_;
  }

  void Backward()
  {
    for (int i = 0; i < mat1_->n_; i++)
    {  // loop over rows of m1
      for (int j = 0; j < mat2_->d_; j++)
      {  // loop over cols of m2
        for (int k = 0; k < mat1_->d_; k++)
        {  // dot product loop
          float b = out_->dw_[mat2_->d_ * i + j];
          mat1_->dw_[mat1_->d_ * i + k] += mat2_->w_[mat2_->d_ * k + j] * b;
          mat2_->dw_[mat2_->d_ * k + j] += mat1_->w_[mat1_->d_ * i + k] * b;
        }
      }
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
