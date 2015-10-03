#ifndef CONV_H
#define CONV_H

#include "utils.h"

class ConvOp : public Object
{
 public:
  ConvOp(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> &filters,
         std::shared_ptr<MatWdw> *out, int filter_width, int filter_height,
         int num_output_channels, int padding_x, int padding_y)
  {
    in_ = in;
    filters_ = filters;

    int stride_x = 1;
    int stride_y = 1;
    params.padding_x = padding_x;
    params.padding_y = padding_y;
    params.stride_x = stride_x;
    params.stride_y = stride_y;
    params.filter_width = filter_width;
    params.filter_height = filter_height;
    params.num_input_channels = in_->size_[2];
    params.num_output_channels = num_output_channels;

    int out_width =
        (in_->size_[0] + 2 * padding_x - filter_width) / stride_x + 1;
    int out_height =
        (in_->size_[1] + 2 * padding_y - filter_height) / stride_y + 1;

    out_ = std::shared_ptr<MatWdw>(
        new MatWdw(out_width, out_height, num_output_channels, in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->Conv(in_->w_, filters_->w_, out_->w_, params);

    return out_;
  }

  void Backward()
  {
    math->ConvDeriv(in_->w_, in_->dw_, filters_->w_, filters_->dw_, out_->w_, out_->dw_,
                    params);
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
    std::fill(filters_->dw_->data_.begin(), filters_->dw_->data_.end(), 0);
  }

  ConvParams params;
  std::shared_ptr<MatWdw> in_;
  std::shared_ptr<MatWdw> filters_;
  std::shared_ptr<MatWdw> out_;
};

#endif
