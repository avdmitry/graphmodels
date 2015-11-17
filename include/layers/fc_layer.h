#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "utils.h"

class FCLayer : public Object
{
 public:
  FCLayer(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> *out,
          int num_output)
  {
    in_ = in;

    int num_input = in_->size_[0] * in_->size_[1] * in_->size_[2];
    float dev = sqrt(1.0 / num_input);
    filters_ = RandMatGauss(num_output, num_input, 1, 1, 0.0, dev);
    biases_ = std::shared_ptr<Mat>(new Mat(1, 1, num_output));

    // printf("fc out: %u %u %u %u -> %u %u %u %u\n", in_->size_[0],
    //       in_->size_[1], in_->size_[2], in_->size_[3], out_width,
    //       out_height, params.num_output_channels, in_->size_[3]);
    out_ = std::shared_ptr<Mat>(
        new Mat(1, 1, num_output, in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<Mat> Forward()
  {
    std::vector<int> in_size(in_->size_);
    std::vector<int> out_size(out_->size_);
    in_->size_[0] =
        in_->size_[0] * in_->size_[1] * in_->size_[2];
    in_->size_[1] = in_->size_[3];
    out_->size_[0] = out_->size_[2];
    out_->size_[1] = in_->size_[3];
    math->Mul(filters_, in_, out_);
    in_->size_ = in_size;
    out_->size_ = out_size;

    math->Add(biases_, out_, out_);

    return out_;
  }

  void Backward()
  {
    std::vector<int> in_size(in_->size_);
    std::vector<int> out_size(out_->size_);
    in_->size_[0] =
        in_->size_[0] * in_->size_[1] * in_->size_[2];
    in_->size_[1] = in_->size_[3];
    out_->size_[0] = out_->size_[2];
    out_->size_[1] = in_->size_[3];
    math->MulDeriv(filters_, in_, filters_->dw_, in_->dw_, out_->dw_);
    in_->size_ = in_size;
    out_->size_ = out_size;

    math->AddDeriv(biases_->dw_, out_->dw_, out_->dw_);
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

  std::shared_ptr<Mat> in_;
  std::shared_ptr<Mat> filters_;
  std::shared_ptr<Mat> biases_;
  std::shared_ptr<Mat> out_;
};

#endif
