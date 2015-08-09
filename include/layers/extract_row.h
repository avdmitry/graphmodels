#ifndef EXTRACT_ROW_H
#define EXTRACT_ROW_H

#include "utils.h"

// Extract row with index idx (optimization for special case MulOp: mat*vec).
class ExtractRowOp : public Object
{
 public:
  ExtractRowOp(std::shared_ptr<MatWdw> &mat, int idx,
               std::shared_ptr<MatWdw> *out)
  {
    assert(idx >= 0 && idx < mat->size_[0]);
    mat_ = mat;
    idx_ = idx;
    out_ = std::shared_ptr<MatWdw>(new MatWdw(mat_->size_[1], 1));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    for (int i = 0; i < mat_->size_[1]; i++)
    {
      out_->w_->data_[i] = mat_->w_->data_[mat_->size_[1] * idx_ + i];
    }

    return out_;
  }

  void Backward()
  {
    for (int i = 0; i < mat_->size_[1]; i++)
    {
      mat_->dw_->data_[mat_->size_[1] * idx_ + i] += out_->dw_->data_[i];
    }
  }

  void ClearDw()
  {
    std::fill(mat_->dw_->data_.begin(), mat_->dw_->data_.end(), 0);
  }

  int idx_;
  std::shared_ptr<MatWdw> mat_;
  std::shared_ptr<MatWdw> out_;
};

#endif
