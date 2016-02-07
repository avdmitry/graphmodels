#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "utils.h"

class BatchNormOp : public Operation
{
 public:
  BatchNormOp(std::string name, std::shared_ptr<Mat> &in,
              std::shared_ptr<Mat> *out)
  {
    name_ = name;
    in_ = in;

    scale_ = std::shared_ptr<Mat>(new Mat(1, 1, in_->size_[2], 1));
    std::fill(scale_->data_.begin(), scale_->data_.end(), 1);
    math->MemoryAlloc(scale_);
    math->MemoryAlloc(scale_->dw_);
    math->CopyToDevice(scale_);

    bias_ = std::shared_ptr<Mat>(new Mat(1, 1, in_->size_[2], 1));
    math->MemoryAlloc(bias_);
    math->MemoryAlloc(bias_->dw_);

    mean_ = std::shared_ptr<Mat>(new Mat(1, 1, in_->size_[2], 1));
    math->MemoryAlloc(mean_);
    math->MemoryAlloc(mean_->dw_);
    variance_ = std::shared_ptr<Mat>(new Mat(1, 1, in_->size_[2], 1));
    math->MemoryAlloc(variance_);
    math->MemoryAlloc(variance_->dw_);

    out_ = std::shared_ptr<Mat>(new Mat(in_->size_));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->BatchNormSetUp(in_, params_);
  }

  void Forward(bool train)
  {
    math->BatchNorm(in_, scale_, bias_, mean_, variance_, out_, params_, train);
  }

  void Backward()
  {
    math->BatchNormDeriv(in_, scale_, bias_, mean_, variance_, out_, params_);
  }

  void SetBatchSize(int new_size)
  {
    in_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
    std::fill(scale_->dw_->data_.begin(), scale_->dw_->data_.end(), 0);
    std::fill(bias_->dw_->data_.begin(), bias_->dw_->data_.end(), 0);
    math->MemoryClear(in_->dw_);
    math->MemoryClear(scale_->dw_);
    math->MemoryClear(bias_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
    params.emplace_back(scale_);
    params.emplace_back(bias_);

    params.emplace_back(mean_);
    params.emplace_back(variance_);
  }

  Params params_;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> scale_;
  std::shared_ptr<Mat> bias_;
  std::shared_ptr<Mat> mean_;
  std::shared_ptr<Mat> variance_;
  std::shared_ptr<Mat> out_;
};

#endif
