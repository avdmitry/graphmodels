#include "utils.h"

#include <random>

using std::string;
using std::vector;
using std::shared_ptr;

shared_ptr<std::default_random_engine> engine(new std::default_random_engine);

shared_ptr<MatWdw> RandMat(int n, int d, float l, float r)
{
  shared_ptr<MatWdw> mat(new MatWdw(n, d));

  for (int i = 0; i < mat->w_->data_.size(); ++i)
  {
    mat->w_->data_[i] = Randf(l, r);
  }

  return mat;
}

shared_ptr<MatWdw> RandMatGauss(int n, int d, int m, int f, float mean,
                                float stddev)
{
  shared_ptr<MatWdw> mat(new MatWdw(n, d, m, f));

  std::normal_distribution<float> distribution(mean, stddev);
  for (int i = 0; i < mat->w_->data_.size(); ++i)
  {
    mat->w_->data_[i] = distribution(*engine);
  }

  return mat;
}

// Argmax of array.
int MaxIdx(const vector<float> &w)
{
  float max_value = w[0];
  int max_idx = 0;
  for (int i = 1; i < w.size(); ++i)
  {
    float value = w[i];
    if (value > max_value)
    {
      max_idx = i;
      max_value = value;
    }
  }
  return max_idx;
}

// Sample argmax from w, assuming w are probabilities that sum to one.
int SampleIdx(vector<float> &w)
{
  float r = Randf(0, 1);
  float x = 0.0;
  for (int i = 0; i < w.size(); ++i)
  {
    x += w[i];
    if (x > r)
    {
      return i;
    }
  }

  return w.size() - 1;  // Should never get here.
}

void Trim(string *str)
{
  // trim leading spaces
  size_t startpos = str->find_first_not_of(" \t");
  if (string::npos != startpos)
  {
    *str = str->substr(startpos);
  }

  // trim trailing spaces
  size_t endpos = str->find_last_not_of(" \t");
  if (string::npos != endpos)
  {
    *str = str->substr(0, endpos + 1);
  }
}

float SoftmaxLoss(shared_ptr<Model> &net, int idx_target)
{
  shared_ptr<MatWdw> &logprobs = net->output_;
  shared_ptr<Mat> probs = math->Softmax(logprobs->w_);
  logprobs->dw_->data_ = probs->data_;
  logprobs->dw_->data_[idx_target] -= 1;
  return -log(probs->data_[idx_target]);
}
