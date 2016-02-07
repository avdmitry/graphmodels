#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "utils.h"

class ConvLayer : public Operation
{
 public:
  ConvLayer(std::string name, std::shared_ptr<Mat> &in,
            std::shared_ptr<Mat> *out, int num_filters, int filter_width,
            int filter_height, int padding_x, int padding_y, int stride_x,
            int stride_y)
  {
    name_ = name;
    in_ = in;

    params_.padding_x = padding_x;
    params_.padding_y = padding_y;
    params_.stride_x = stride_x;
    params_.stride_y = stride_y;
    params_.filter_width = filter_width;
    params_.filter_height = filter_height;
    params_.num_input = in_->size_[2];
    params_.num_output = num_filters;
    params_.out_width =
        (in_->size_[0] + 2 * padding_x - filter_width) / stride_x + 1;
    params_.out_height =
        (in_->size_[1] + 2 * padding_y - filter_height) / stride_y + 1;

    float dev = sqrt(1.0 / (filter_width * filter_height * params_.num_input));
    filters_ = RandMatGauss(filter_width, filter_height, params_.num_input,
                            params_.num_output, 0.0, dev);
    math->MemoryAlloc(filters_);
    math->MemoryAlloc(filters_->dw_);

    biases_ = std::shared_ptr<Mat>(new Mat(1, 1, params_.num_output));
    std::fill(biases_->data_.begin(), biases_->data_.end(), 0.0);
    math->MemoryAlloc(biases_);
    math->MemoryAlloc(biases_->dw_);

    printf("conv out: %u %u %u %u -> %u %u %u %u\n", in_->size_[0],
           in_->size_[1], in_->size_[2], in_->size_[3], params_.out_width,
           params_.out_height, params_.num_output, in_->size_[3]);
    out_ = std::shared_ptr<Mat>(new Mat(params_.out_width, params_.out_height,
                                        params_.num_output, in_->size_[3]));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->ConvSetUp(in_, params_);

    math->CopyToDevice(filters_);
    math->CopyToDevice(biases_);
  }

  void Forward(bool train)
  {
    math->Conv(in_, filters_, biases_, out_, params_);
  }

  void Backward()
  {
    math->ConvDeriv(in_, filters_, biases_, out_, params_);
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

  Params params_;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> filters_;
  std::shared_ptr<Mat> biases_;
  std::shared_ptr<Mat> out_;
};

#endif
