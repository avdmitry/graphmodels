#ifndef MUL_H
#define MUL_H

#include "utils.h"

// Multiply matrices.
class MulOp : public Operation
{
 public:
  MulOp(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
        std::shared_ptr<Mat> *out)
  {
    assert(mat1->size_[1] == mat2->size_[0]);
    mat1_ = mat1;
    mat2_ = mat2;
    out_ = std::shared_ptr<Mat>(new Mat(mat1_->size_[0], mat2_->size_[1],
                                        mat1_->size_[2], mat1_->size_[3]));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->MemoryAlloc(mat1_);
    math->MemoryAlloc(mat1_->dw_);
    math->MemoryAlloc(mat2_);
    math->MemoryAlloc(mat2_->dw_);
  }

  void Forward(bool train)
  {
    math->Mul(mat1_, mat2_, out_);
  }

  void Backward()
  {
    math->MulDeriv(mat1_, mat2_, mat1_->dw_, mat2_->dw_, out_->dw_);
  }

  void SetBatchSize(int new_size)
  {
    mat1_->size_[3] = new_size;
    mat2_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(mat1_->dw_->data_.begin(), mat1_->dw_->data_.end(), 0);
    std::fill(mat2_->dw_->data_.begin(), mat2_->dw_->data_.end(), 0);
    math->MemoryClear(mat1_->dw_);
    math->MemoryClear(mat2_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  std::shared_ptr<Mat> mat1_, mat2_;
  std::shared_ptr<Mat> out_;
};

#endif
