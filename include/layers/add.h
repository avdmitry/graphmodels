#ifndef ADD_H
#define ADD_H

#include "utils.h"

// Add matrices.
class AddOp : public Object
{
 public:
  AddOp(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
        std::shared_ptr<Mat> *out)
  {
    // printf("%lu %lu\n", mat1->data_.size(), mat2->data_.size());
    assert(mat1->data_.size() == mat2->data_.size());
    mat1_ = mat1;
    mat2_ = mat2;
    out_ = std::shared_ptr<Mat>(new Mat(mat1_->size_));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    math->Add(mat1_, mat2_, out_);

    return out_;
  }

  void Backward()
  {
    math->AddDeriv(mat1_->dw_, mat2_->dw_, out_->dw_);
  }

  void ClearDw()
  {
    std::fill(mat1_->dw_->data_.begin(), mat1_->dw_->data_.end(), 0);
    std::fill(mat2_->dw_->data_.begin(), mat2_->dw_->data_.end(), 0);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  std::shared_ptr<Mat> mat1_, mat2_;
  std::shared_ptr<Mat> out_;
};

#endif
