#ifndef MAX_POOL_H
#define MAX_POOL_H

#include "utils.h"

class MaxPoolOp : public Object
{
 public:
  MaxPoolOp(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out,
            int filter_width, int filter_height)
  {
    in_ = in;

    params.padding_x = 1;
    params.padding_y = 1;
    params.stride_x = 2;
    params.stride_y = 2;
    params.filter_width = filter_width;
    params.filter_height = filter_height;
    params.num_output_channels = in_->size_[2];

    int out_width = (in_->size_[0] + params.padding_x * 2 - filter_width) /
                        params.stride_x +
                    1;
    int out_height = (in_->size_[1] + params.padding_y * 2 - filter_height) /
                         params.stride_y +
                     1;
printf("%u %u %u %u\n", out_width, out_height, in_->size_[2], in_->size_[3]);
    out_ = std::shared_ptr<MatWdw>(
        new MatWdw(out_width, out_height, in_->size_[2], in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    math->MaxPool(in_->w_, out_->w_, params);

    return out_;
  }

  void Backward()
  {
    math->MaxPoolDeriv(in_->w_, in_->dw_, out_->w_, out_->dw_, params);
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
  }

  ConvParams params;
  std::shared_ptr<MatWdw> in_;
  std::shared_ptr<MatWdw> out_;
};

#endif
