#include "utils.h"

#include <random>
#include <sstream>

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

shared_ptr<vector<string>> Split(const string &s, char delim)
{
  shared_ptr<vector<string>> elems(new vector<string>);
  std::stringstream ss(s);
  string item;
  while (getline(ss, item, delim))
  {
    elems->push_back(item);
  }
  return elems;
}
