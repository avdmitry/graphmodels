#ifndef ExtractRow_H
#define ExtractRow_H

#include "utils.h"

// Extract row with index idx.
class ExtractRowOp : public Object
{
 public:
  ExtractRowOp(std::shared_ptr<Mat> &mat, int idx, std::shared_ptr<Mat> *out)
  {
    assert(idx >= 0 && idx < mat->n_);
    mat_ = mat;
    idx_ = idx;
    out_ = std::shared_ptr<Mat>(new Mat(mat_->d_, 1));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    for (int i = 0; i < mat_->d_; i++)
    {
      out_->w_[i] = mat_->w_[mat_->d_ * idx_ + i];
    }

    return out_;
  }

  void Backward()
  {
    for (int i = 0; i < mat_->d_; i++)
    {
      mat_->dw_[mat_->d_ * idx_ + i] += out_->dw_[i];
    }
  }

  int idx_;
  std::shared_ptr<Mat> mat_;
  std::shared_ptr<Mat> out_;
};

#endif
