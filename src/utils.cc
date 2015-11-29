#include "utils.h"

#include <random>

using std::string;
using std::vector;
using std::shared_ptr;

shared_ptr<std::default_random_engine> engine(new std::default_random_engine);

shared_ptr<Mat> RandMat(int n, int d, float l, float r)
{
  shared_ptr<Mat> mat(new Mat(n, d));

  for (int i = 0; i < mat->data_.size(); ++i)
  {
    mat->data_[i] = Randf(l, r);
  }

  return mat;
}

shared_ptr<Mat> RandMatGauss(int n, int d, int m, int f, float mean,
                             float stddev)
{
  shared_ptr<Mat> mat(new Mat(n, d, m, f));

  std::normal_distribution<float> distribution(mean, stddev);
  for (int i = 0; i < mat->data_.size(); ++i)
  {
    mat->data_[i] = distribution(*engine);
  }

  return mat;
}

// Argmax of array.
int MaxIdx(const shared_ptr<Mat> &mat)
{
  float max_value = mat->data_[0];
  int max_idx = 0;
  int size = mat->size_[0] * mat->size_[1] * mat->size_[2] * mat->size_[3];
  for (int i = 1; i < size; ++i)
  {
    float value = mat->data_[i];
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
  float r = Random01();
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

float SoftmaxLoss(shared_ptr<Model> &net, vector<int> &idx_target)
{
  shared_ptr<Mat> &logprobs = net->output_;
  shared_ptr<Mat> probs = math->Softmax(logprobs);
  logprobs->dw_->data_ = probs->data_;

  float loss = 0;
  int num_elements = probs->size_[0] * probs->size_[1] * probs->size_[2];
  for (int batch = 0; batch < probs->size_[3]; ++batch)
  {
    int idx = batch * num_elements + idx_target[batch];
    logprobs->dw_->data_[idx] -= 1;
    loss -= log(probs->data_[idx]);
  }

  return loss;
}
