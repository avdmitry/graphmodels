#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "utils.h"

class FCLayer : public Object
{
 public:
  FCLayer(std::shared_ptr<MatWdw> &in, std::shared_ptr<MatWdw> *out,
          int num_output)
  {
    in_ = in;

    int num_input = in_->size_[0] * in_->size_[1] * in_->size_[2];
    float dev = sqrt(1.0 / num_input);
    filters_ = RandMatGauss(num_output, num_input, 1, 1, 0.0, dev);
    biases_ = std::shared_ptr<MatWdw>(new MatWdw(1, 1, num_output));

    // printf("fc out: %u %u %u %u -> %u %u %u %u\n", in_->size_[0],
    //       in_->size_[1], in_->size_[2], in_->size_[3], out_width, out_height,
    //      params.num_output_channels, in_->size_[3]);
    out_ = std::shared_ptr<MatWdw>(new MatWdw(1, 1, num_output, in_->size_[3]));
    *out = out_;
  }

  std::shared_ptr<MatWdw> Forward()
  {
    int num_out = out_->size_[2];
    for (int i = 0; i < num_out; ++i)
    {
      float result = biases_->w_->data_[i];
      int num_in = filters_->size_[1];
      int offset = num_in * i;
      for (int j = 0; j < num_in; ++j)
      {
        result += in_->w_->data_[j] * filters_->w_->data_[offset + j];
      }
      out_->w_->data_[i] = result;
    }

    return out_;
  }

  void Backward()
  {
    int num_out = out_->size_[2];
    for (int i = 0; i < num_out; ++i)
    {
      float dw = out_->dw_->data_[i];
      int num_in = filters_->size_[1];
      int offset = num_in * i;
      for (int j = 0; j < num_in; ++j)
      {
        in_->dw_->data_[j] += dw * filters_->w_->data_[offset + j];
        filters_->dw_->data_[offset + j] += dw * in_->w_->data_[j];
      }
      biases_->dw_->data_[i] += dw;
    }
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

  std::shared_ptr<MatWdw> in_;
  std::shared_ptr<MatWdw> filters_;
  std::shared_ptr<MatWdw> biases_;
  std::shared_ptr<MatWdw> out_;
};

#endif
