#ifndef RNN_H
#define RNN_H

#include "utils.h"
#include "layers.h"

class rnn : public Model
{
 public:
  rnn(int input_size, std::vector<int> hidden_sizes, int output_size);

  std::shared_ptr<Mat> Forward(std::shared_ptr<Graph> &graph, int idx);

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
