#ifndef RNN_H
#define RNN_H

#include "utils.h"
#include "layers.h"

class Rnn : public Model
{
 public:
  Rnn(int input_size, std::vector<int> hidden_sizes, int output_size);

  void Create(int idx);

  void GetParameters(std::vector<std::shared_ptr<MatWdw>> &params_);

  void ClearPrevState()
  {
    prev_hiddens_.clear();
  }

  std::vector<std::shared_ptr<MatWdw>> wxh_, whh_, bhh_;
  std::shared_ptr<MatWdw> whd_, bd_;
  std::shared_ptr<MatWdw> wil_;

  std::vector<std::shared_ptr<MatWdw>> prev_hiddens_;

  std::vector<int> hidden_sizes_;
};

#endif
