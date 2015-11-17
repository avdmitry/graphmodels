#ifndef RNN_H
#define RNN_H

#include "utils.h"
#include "layers.h"

class Rnn : public Model
{
 public:
  Rnn(int input_size, std::vector<int> hidden_sizes, int output_size);

  void Create(int idx);

  void GetParameters(std::vector<std::shared_ptr<Mat>> &params_);

  void ClearPrevState()
  {
    prev_hiddens_.clear();
  }

  std::vector<std::shared_ptr<Mat>> wxh_, whh_, bhh_;
  std::shared_ptr<Mat> whd_, bd_;
  std::shared_ptr<Mat> wil_;

  std::vector<std::shared_ptr<Mat>> prev_hiddens_;

  std::vector<int> hidden_sizes_;
};

#endif
