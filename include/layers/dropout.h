#ifndef DROPOUT_H
#define DROPOUT_H

#include "utils.h"

class DropoutOp : public Operation
{
 public:
  DropoutOp(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out, float prob)
  {
    in_ = in;
    mask_ = std::shared_ptr<Mat>(new Mat(in_->size_));
    prob_ = prob;
    out_ = std::shared_ptr<Mat>(new Mat(in_->size_));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;
  }

  void Forward(bool train)
  {
    if (!train)
    {
      out_->data_ = in_->data_;
      return;
    }

    float scale = 1.0 / (1.0 - prob_);
    for (int i = 0; i < mask_->data_.size(); ++i)
    {
      if (Random01() < prob_)
      {
        mask_->data_[i] = 0;
      }
      else
      {
        mask_->data_[i] = scale;
      }
    }

    math->ElmtMul(in_, mask_, out_);
  }

  void Backward()
  {
    math->ElmtMulDeriv(in_, mask_, in_->dw_, mask_->dw_, out_->dw_);
  }

  void SetBatchSize(int new_size)
  {
    in_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
    std::fill(mask_->dw_->data_.begin(), mask_->dw_->data_.end(), 0);
    math->MemoryClear(in_->dw_);
    math->MemoryClear(mask_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> out_;
  std::shared_ptr<Mat> mask_;
  float prob_;
};

#endif
