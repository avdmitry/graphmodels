#ifndef TANH_H
#define TANH_H

#include "utils.h"

class TanhOp : public Operation
{
 public:
  TanhOp(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out)
  {
    in_ = in;
    out_ = std::shared_ptr<Mat>(new Mat(in_->size_));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->ActivSetUp(in_, params_);
  }

  void Forward(bool train)
  {
    math->Tanh(in_, out_, params_);
  }

  void Backward()
  {
    math->TanhDeriv(in_, out_, params_);
  }

  void SetBatchSize(int new_size)
  {
    in_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
    math->MemoryClear(in_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  Params params_;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> out_;
};

#endif
