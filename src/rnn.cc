#include "rnn.h"

using namespace std;

Rnn::Rnn(int input_size, vector<int> hidden_sizes, int output_size)
    : hidden_sizes_(hidden_sizes)
{
  int hidden_size;
  for (size_t d = 0; d < hidden_sizes_.size(); d++)
  {
    int prev_size;
    if (d == 0)
    {
      prev_size = input_size;
    }
    else
    {
      prev_size = hidden_sizes_[d - 1];
    }
    hidden_size = hidden_sizes_[d];

    wxh_.emplace_back(RandMat(hidden_size, prev_size, -0.08, 0.08));
    whh_.emplace_back(RandMat(hidden_size, hidden_size, -0.08, 0.08));
    bhh_.emplace_back(new MatWdw(hidden_size, 1));
  }

  // Decoder params.
  whd_ = RandMat(output_size, hidden_size, -0.08, 0.08);
  bd_ = shared_ptr<MatWdw>(new MatWdw(output_size, 1));

  wil_ = RandMat(input_size, output_size, -0.08, 0.08);

  GetParameters(params_);
  for (size_t i = 0; i < params_.size(); ++i)
  {
    shared_ptr<MatWdw> &mat = params_[i];
    params_prev_.emplace_back(new MatWdw(mat->size_[0], mat->size_[1]));
  }
}

void Rnn::Create(shared_ptr<Graph> &graph, int idx)
{
  if (prev_hiddens_.size() == 0)
  {
    for (size_t d = 0; d < hidden_sizes_.size(); d++)
    {
      prev_hiddens_.emplace_back(new MatWdw(hidden_sizes_[d], 1));
    }
  }

  input_ = shared_ptr<MatWdw>(new MatWdw(wil_->size_[1], 1));
  fill(input_->w_->data_.begin(), input_->w_->data_.end(), 0);
  input_->w_->data_[idx] = 1;

  shared_ptr<MatWdw> x;
  graph->Process(shared_ptr<Object>(new MulOp(wil_, input_, &x)));

  vector<shared_ptr<MatWdw>> hidden;
  for (size_t d = 0; d < hidden_sizes_.size(); d++)
  {
    shared_ptr<MatWdw> input_vector;
    if (d == 0)
    {
      input_vector = x;
    }
    else
    {
      input_vector = hidden[d - 1];
    }
    shared_ptr<MatWdw> &hidden_prev = prev_hiddens_[d];

    shared_ptr<MatWdw> h0, h1, h01, bias, hidden_curr;
    graph->Process(shared_ptr<Object>(new MulOp(wxh_[d], input_vector, &h0)));
    graph->Process(shared_ptr<Object>(new MulOp(whh_[d], hidden_prev, &h1)));
    graph->Process(shared_ptr<Object>(new AddOp(h0, h1, &h01)));
    graph->Process(shared_ptr<Object>(new AddOp(h01, bhh_[d], &bias)));
    graph->Process(shared_ptr<Object>(new ReluOp(bias, &hidden_curr)));

    hidden.emplace_back(hidden_curr);
  }

  // Decoder.
  shared_ptr<MatWdw> hd;
  graph->Process(
      shared_ptr<Object>(new MulOp(whd_, hidden[hidden.size() - 1], &hd)));
  graph->Process(shared_ptr<Object>(new AddOp(hd, bd_, &output_)));

  prev_hiddens_ = hidden;
}

void Rnn::GetParameters(vector<shared_ptr<MatWdw>> &params)
{
  for (size_t i = 0; i < hidden_sizes_.size(); ++i)
  {
    params.emplace_back(wxh_[i]);
    params.emplace_back(whh_[i]);
    params.emplace_back(bhh_[i]);
  }
  params.emplace_back(whd_);
  params.emplace_back(bd_);
  params.emplace_back(wil_);
}
