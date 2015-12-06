#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "utils.h"

class FCLayer : public Operation
{
 public:
  FCLayer(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out, int num_output)
  {
    in_ = in;

    int num_input = in_->size_[0] * in_->size_[1] * in_->size_[2];
    float dev = sqrt(1.0 / num_input);
    filters_ = RandMatGauss(num_input, num_output, 1, 1, 0.0, dev);
    math->MemoryAlloc(filters_);
    math->MemoryAlloc(filters_->dw_);

    biases_ = std::shared_ptr<Mat>(new Mat(1, 1, num_output));
    math->MemoryAlloc(biases_);
    math->MemoryAlloc(biases_->dw_);

    // printf("fc out: %u %u %u %u -> %u %u %u %u\n", in_->size_[0],
    //       in_->size_[1], in_->size_[2], in_->size_[3], out_width,
    //       out_height, params.num_output_channels, in_->size_[3]);
    out_ = std::shared_ptr<Mat>(new Mat(1, 1, num_output, in_->size_[3]));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->CopyToDevice(filters_);
    math->CopyToDevice(biases_);
  }

  void Forward(bool train)
  {
    math->Fc(in_, filters_, biases_, out_);
  }

  void Backward()
  {
    math->FcDeriv(in_, filters_, biases_, out_);
  }

  void SetBatchSize(int new_size)
  {
    in_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
    std::fill(filters_->dw_->data_.begin(), filters_->dw_->data_.end(), 0);
    std::fill(biases_->dw_->data_.begin(), biases_->dw_->data_.end(), 0);
    math->MemoryClear(in_->dw_);
    math->MemoryClear(filters_->dw_);
    math->MemoryClear(biases_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
    params.emplace_back(filters_);
    params.emplace_back(biases_);
  }

  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> filters_;
  std::shared_ptr<Mat> biases_;
  std::shared_ptr<Mat> out_;
};

#endif
