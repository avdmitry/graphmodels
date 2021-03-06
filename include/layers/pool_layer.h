#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include "utils.h"

class PoolLayer : public Operation
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
    params_.num_output = in_->size_[2];
    params_.out_width =
        (in_->size_[0] + padding_x * 2 - filter_width) / stride_x + 1;
    params_.out_height =
        (in_->size_[1] + padding_y * 2 - filter_height) / stride_y + 1;

    printf("pool out: %u %u %u %u\n", params_.out_width, params_.out_height,
           in_->size_[2], in_->size_[3]);
    out_ = std::shared_ptr<Mat>(new Mat(params_.out_width, params_.out_height,
                                        in_->size_[2], in_->size_[3]));
    math->MemoryAlloc(out_);
    math->MemoryAlloc(out_->dw_);
    *out = out_;

    math->PoolSetUp(in_, type_, params_);
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
        math->MaxPoolDeriv(in_, out_, params_);
        break;
      case AVE:
        math->AvePoolDeriv(in_, out_, params_);
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
    math->MemoryClear(in_->dw_);
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
  }

  Params params_;
  PoolType type_;
  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> out_;
};

#endif
