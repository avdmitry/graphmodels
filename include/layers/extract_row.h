#ifndef EXTRACT_ROW_H
#define EXTRACT_ROW_H

#include "utils.h"

// Extract row with index idx (optimization for special case MulOp: mat*vec).
class ExtractRowOp : public Operation
{
 public:
  ExtractRowOp(std::shared_ptr<Mat> &mat, int idx, std::shared_ptr<Mat> *out)
  {
    assert(idx >= 0 && idx < mat->size_[0]);
    mat_ = mat;
    idx_ = idx;
    out_ = std::shared_ptr<Mat>(new Mat(mat_->size_[1], 1));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->MemoryAlloc(mat_);
    math->MemoryAlloc(mat_->dw_);
  }

  void Forward(bool train)
  {
    for (int i = 0; i < mat_->size_[1]; i++)
    {
      out_->data_[i] = mat_->data_[mat_->size_[1] * idx_ + i];
    }
  }

  void Backward()
  {
    for (int i = 0; i < mat_->size_[1]; i++)
    {
      mat_->dw_->data_[mat_->size_[1] * idx_ + i] += out_->dw_->data_[i];
    }
  }

  void SetBatchSize(int new_size)
  {
    mat_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(mat_->dw_->data_.begin(), mat_->dw_->data_.end(), 0);
    math->MemoryClear(mat_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  int idx_;
  std::shared_ptr<Mat> mat_;
  std::shared_ptr<Mat> out_;
};

#endif
