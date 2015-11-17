#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H

#include "utils.h"

class MaxPoolLayer : public Object
{
 public:
  MaxPoolLayer(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out,
               int filter_width, int filter_height, int padding_x,
               int padding_y, int stride_x, int stride_y)
  {
    in_ = in;

    params.padding_x = padding_x;
    params.padding_y = padding_y;
    params.stride_x = stride_x;
    params.stride_y = stride_y;
    params.filter_width = filter_width;
    params.filter_height = filter_height;
    params.num_output_channels = in_->size_[2];

    int out_width = (in_->size_[0] + params.padding_x * 2 - filter_width) /
                        params.stride_x +
                    1;
    int out_height =
        (in_->size_[1] + params.padding_y * 2 - filter_height) /
            params.stride_y +
        1;
    printf("maxpool out: %u %u %u %u\n", out_width, out_height,
           in_->size_[2], in_->size_[3]);
    out_ = std::shared_ptr<Mat>(new Mat(
        out_width, out_height, in_->size_[2], in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    math->MaxPool(in_, out_, params);

    return out_;
  }

  void Backward()
  {
    math->MaxPoolDeriv(in_, in_->dw_, out_, out_->dw_, params);
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  ConvParams params;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> out_;
};

#endif
