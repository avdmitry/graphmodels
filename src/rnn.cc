#include "rnn.h"

using std::string;
using std::vector;
using std::shared_ptr;

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
    bhh_.emplace_back(new Mat(hidden_size, 1));
  }

  // Decoder params.
  whd_ = RandMat(output_size, hidden_size, -0.08, 0.08);
  bd_ = shared_ptr<Mat>(new Mat(output_size, 1));

  wil_ = RandMat(input_size, output_size, -0.08, 0.08);

  GetParameters(params_);
  for (size_t i = 0; i < params_.size(); ++i)
  {
    shared_ptr<Mat> &mat = params_[i];
    params_prev_.emplace_back(new Mat(mat->size_));
  }
}

void Rnn::Create(int idx)
{
  if (prev_hiddens_.size() == 0)
  {
    for (size_t d = 0; d < hidden_sizes_.size(); d++)
    {
      prev_hiddens_.emplace_back(new Mat(hidden_sizes_[d], 1));
    }
  }

  input_ = shared_ptr<Mat>(new Mat(wil_->size_[1], 1));
  fill(input_->data_.begin(), input_->data_.end(), 0);
  input_->data_[idx] = 1;

  shared_ptr<Mat> x;
  graph_->Process(shared_ptr<Object>(new MulOp(wil_, input_, &x)));

  vector<shared_ptr<Mat>> hidden;
  for (size_t d = 0; d < hidden_sizes_.size(); d++)
  {
    shared_ptr<Mat> input_vector;
    if (d == 0)
    {
      input_vector = x;
    }
    else
    {
      input_vector = hidden[d - 1];
    }
    shared_ptr<Mat> &hidden_prev = prev_hiddens_[d];

    shared_ptr<Mat> h0, h1, h01, bias, hidden_curr;
    graph_->Process(shared_ptr<Object>(new MulOp(wxh_[d], input_vector, &h0)));
    graph_->Process(shared_ptr<Object>(new MulOp(whh_[d], hidden_prev, &h1)));
    graph_->Process(shared_ptr<Object>(new AddOp(h0, h1, &h01)));
    graph_->Process(shared_ptr<Object>(new AddOp(h01, bhh_[d], &bias)));
    graph_->Process(shared_ptr<Object>(new ReluOp(bias, &hidden_curr)));

    hidden.emplace_back(hidden_curr);
  }

  // Decoder.
  shared_ptr<Mat> hd;
  graph_->Process(
      shared_ptr<Object>(new MulOp(whd_, hidden[hidden.size() - 1], &hd)));
  graph_->Process(shared_ptr<Object>(new AddOp(hd, bd_, &output_)));

  prev_hiddens_ = hidden;
}

void Rnn::GetParameters(vector<shared_ptr<Mat>> &params)
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
