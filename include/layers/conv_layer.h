#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "utils.h"

class ConvLayer : public Object
{
 public:
  ConvLayer(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out,
            int num_filters, int filter_width, int filter_height, int padding_x,
            int padding_y, int stride_x, int stride_y)
  {
    in_ = in;

    params_.padding_x = padding_x;
    params_.padding_y = padding_y;
    params_.stride_x = stride_x;
    params_.stride_y = stride_y;
    params_.filter_width = filter_width;
    params_.filter_height = filter_height;
    params_.num_input_channels = in_->size_[2];
    params_.num_output_channels = num_filters;

    float dev =
        sqrt(1.0 / (filter_width * filter_height * params_.num_input_channels));
    filters_ =
        RandMatGauss(filter_width, filter_height, params_.num_input_channels,
                     params_.num_output_channels, 0.0, dev);
    biases_ = std::shared_ptr<Mat>(new Mat(1, 1, params_.num_output_channels));
    std::fill(biases_->data_.begin(), biases_->data_.end(), 0.0);

    int out_width =
        (in_->size_[0] + 2 * padding_x - params_.filter_width) / stride_x + 1;
    int out_height =
        (in_->size_[1] + 2 * padding_y - params_.filter_height) / stride_y + 1;

    printf("conv out: %u %u %u %u -> %u %u %u %u\n", in_->size_[0],
           in_->size_[1], in_->size_[2], in_->size_[3], out_width, out_height,
           params_.num_output_channels, in_->size_[3]);
    out_ = std::shared_ptr<Mat>(new Mat(
        out_width, out_height, params_.num_output_channels, in_->size_[3]));
    *out = out_;
  }

  void Forward(bool train)
  {
    math->Conv(in_, filters_, biases_, out_, params_);
  }

  void Backward()
  {
    math->ConvDeriv(in_, in_->dw_, filters_, filters_->dw_, biases_->dw_, out_,
                    out_->dw_, params_);
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
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
    params.emplace_back(filters_);
    params.emplace_back(biases_);
  }

  ConvParams params_;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> filters_;
  std::shared_ptr<Mat> biases_;
  std::shared_ptr<Mat> out_;
};

#endif
