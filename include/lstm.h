#ifndef LSTM_H
#define LSTM_H

#include "utils.h"
#include "layers.h"

class Lstm : public Model
{
 public:
  Lstm(int input_size, std::vector<int> hidden_sizes, int output_size);

  void Create(std::shared_ptr<Graph> &graph, int idx);

  void GetParameters(std::vector<std::shared_ptr<MatWdw>> &params_);

  void ClearPrevState()
  {
    prev_hiddens_.clear();
    prev_cells_.clear();
  }

  std::vector<std::shared_ptr<MatWdw>> wix_, wih_, bi_, wfx_, wfh_, bf_;
  std::vector<std::shared_ptr<MatWdw>> wox_, woh_, bo_, wcx_, wch_, bc_;
  std::shared_ptr<MatWdw> whd_, bd_;
  std::shared_ptr<MatWdw> wil_;

  std::vector<std::shared_ptr<MatWdw>> prev_hiddens_;
  std::vector<std::shared_ptr<MatWdw>> prev_cells_;

  std::vector<int> hidden_sizes_;
};

#endif
