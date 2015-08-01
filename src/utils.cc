#include "utils.h"

#include <random>

using namespace std;

shared_ptr<default_random_engine> engine(new default_random_engine);

shared_ptr<Mat> RandMat(int n, int d, float l, float r)
{
  shared_ptr<Mat> mat(new Mat(n, d));

  for (int i = 0; i < mat->w_.size(); ++i)
  {
    mat->w_[i] = Randf(l, r);
  }

  return mat;
}

shared_ptr<Mat> RandMatGauss(int n, int d, float mean, float stddev)
{
  shared_ptr<Mat> mat(new Mat(n, d));

  normal_distribution<float> distribution(mean, stddev);
  for (int i = 0; i < mat->w_.size(); ++i)
  {
    mat->w_[i] = distribution(*engine);
  }

  return mat;
}

shared_ptr<Mat> Softmax(shared_ptr<Mat> &mat)
{
  shared_ptr<Mat> out(new Mat(mat->n_, mat->d_));
  float maxval = mat->w_[0];
  for (int i = 0; i < mat->w_.size(); i++)
  {
    if (mat->w_[i] > maxval)
    {
      maxval = mat->w_[i];
    }
  }

  float sum = 0.0;
  for (int i = 0; i < out->w_.size(); i++)
  {
    out->w_[i] = exp(mat->w_[i] - maxval);
    sum += out->w_[i];
  }
  for (int i = 0; i < out->w_.size(); i++)
  {
    out->w_[i] /= sum;
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
