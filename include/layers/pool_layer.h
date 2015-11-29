#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include "utils.h"

enum PoolType
{
  MAX = 0,
  AVE = 1
};

class PoolLayer : public Object
{
 public:
  PoolLayer(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out,
            int filter_width, int filter_height, int padding_x, int padding_y,
            int stride_x, int stride_y, PoolType type)
  {
    in_ = in;

    type_ = type;
    params_.padding_x = padding_x;
    params_.padding_y = padding_y;
    params_.stride_x = stride_x;
    params_.stride_y = stride_y;
    params_.filter_width = filter_width;
    params_.filter_height = filter_height;
    params_.num_output_channels = in_->size_[2];

    int out_width = (in_->size_[0] + params_.padding_x * 2 - filter_width) /
                        params_.stride_x +
                    1;
    int out_height = (in_->size_[1] + params_.padding_y * 2 - filter_height) /
                         params_.stride_y +
                     1;
    printf("pool out: %u %u %u %u\n", out_width, out_height, in_->size_[2],
           in_->size_[3]);
    out_ = std::shared_ptr<Mat>(
        new Mat(out_width, out_height, in_->size_[2], in_->size_[3]));
    *out = out_;
  }

  void Forward(bool train)
  {
    switch (type_)
    {
      case MAX:
        math->MaxPool(in_, out_, params_);
        break;
      case AVE:
        math->AvePool(in_, out_, params_);
        break;
    }
  }

  void Backward()
  {
    switch (type_)
    {
      case MAX:
        math->MaxPoolDeriv(in_, in_->dw_, out_, out_->dw_, params_);
        break;
      case AVE:
        math->AvePoolDeriv(in_, in_->dw_, out_, out_->dw_, params_);
        break;
    }
  }

  void SetBatchSize(int new_size)
  {
    in_->size_[3] = new_size;
    out_->size_[3] = new_size;
  }

  void ClearDw()
  {
    std::fill(in_->dw_->data_.begin(), in_->dw_->data_.end(), 0);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  ConvParams params_;
  PoolType type_;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> out_;
};

#endif
