#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "utils.h"

class ConvLayer : public Object
{
 public:
  ConvLayer(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out,
            int num_filters, int filter_width, int filter_height, int padding_x,
            int padding_y, int stride_x, int stride_y)
  {
    in_ = in;

    params.padding_x = padding_x;
    params.padding_y = padding_y;
    params.stride_x = stride_x;
    params.stride_y = stride_y;
    params.filter_width = filter_width;
    params.filter_height = filter_height;
    params.num_input_channels = in_->size_[2];
    params.num_output_channels = num_filters;

    float dev =
        sqrt(1.0 / (filter_width * filter_height * params.num_input_channels));
    filters_ =
        RandMatGauss(filter_width, filter_height, params.num_input_channels,
                     params.num_output_channels, 0.0, dev);
    biases_ =
        std::shared_ptr<MatWdw>(new MatWdw(1, 1, params.num_output_channels));
    std::fill(biases_->w_->data_.begin(), biases_->w_->data_.end(), 0.0);

    int out_width =
        (in_->size_[0] + 2 * padding_x - params.filter_width) / stride_x + 1;
    int out_height =
        (in_->size_[1] + 2 * padding_y - params.filter_height) / stride_y + 1;

    printf("conv out: %u %u %u %u -> %u %u %u %u\n", in_->size_[0],
           in_->size_[1], in_->size_[2], in_->size_[3], out_width, out_height,
           params.num_output_channels, in_->size_[3]);
    out_ = std::shared_ptr<MatWdw>(new MatWdw(
        out_width, out_height, params.num_output_channels, in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Conv(in_->w_, filters_->w_, biases_->w_, out_->w_, params);

    return out_;
  }

  void Backward()
  {
    math->ConvDeriv(in_->w_, in_->dw_, filters_->w_, filters_->dw_,
                    biases_->dw_, out_->w_, out_->dw_, params);
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
    std::fill(filters_->dw_->data_.begin(), filters_->dw_->data_.end(), 0);
    std::fill(biases_->dw_->data_.begin(), biases_->dw_->data_.end(), 0);
  }

  void GetParams(std::vector<std::shared_ptr<MatWdw>> &params)
  {
    params.emplace_back(filters_);
    params.emplace_back(biases_);
  }

  ConvParams params;
  std::shared_ptr<MatWdw> in_;
  std::shared_ptr<MatWdw> filters_;
  std::shared_ptr<MatWdw> biases_;
  std::shared_ptr<MatWdw> out_;
};

#endif
