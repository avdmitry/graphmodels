#include "utils.h"

#include <random>

using namespace std;

shared_ptr<default_random_engine> engine(new default_random_engine);

shared_ptr<MatWdw> RandMat(int n, int d, float l, float r)
{
  shared_ptr<MatWdw> mat(new MatWdw(n, d));

  for (int i = 0; i < mat->w_->data_.size(); ++i)
  {
    mat->w_->data_[i] = Randf(l, r);
  }

  return mat;
}

shared_ptr<MatWdw> RandMatGauss(int n, int d, float mean, float stddev)
{
  shared_ptr<MatWdw> mat(new MatWdw(n, d));

  normal_distribution<float> distribution(mean, stddev);
  for (int i = 0; i < mat->w_->data_.size(); ++i)
  {
    mat->w_->data_[i] = distribution(*engine);
  }

  return mat;
}

shared_ptr<Mat> Softmax(std::shared_ptr<Mat> &mat)
{
  shared_ptr<Mat> out(new Mat(mat->size_[0], mat->size_[1]));
  float maxval = mat->data_[0];
  for (int i = 0; i < mat->data_.size(); i++)
  {
    if (mat->data_[i] > maxval)
    {
      maxval = mat->data_[i];
    }
  }

  float sum = 0.0;
  for (int i = 0; i < out->data_.size(); i++)
  {
    out->data_[i] = exp(mat->data_[i] - maxval);
    sum += out->data_[i];
  }
  for (int i = 0; i < out->data_.size(); i++)
  {
    out->data_[i] /= sum;
  }

  return out;
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
