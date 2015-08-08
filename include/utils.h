#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>   // printf
#include <stdlib.h>  // srand, rand
#include <time.h>    // time
#include <math.h>
#include <assert.h>

#include <vector>
#include <string>
#include <set>
#include <fstream>
#include <map>
#include <memory>

#include "math/common.h"

class MatWdw
{
 public:
  MatWdw()
  {
  }
  ~MatWdw()
  {
  }

  MatWdw(int n, int d)
  {
    // w_ = std::shared_ptr<Mat>(new Mat(n, d));
    // dw_ = std::shared_ptr<Mat>(new Mat(n, d));
    w_ = std::shared_ptr<Mat>(new Mat(std::vector<int>({n, d, 1, 1})));
    dw_ = std::shared_ptr<Mat>(new Mat(std::vector<int>({n, d, 1, 1})));

    size_.emplace_back(n);
    size_.emplace_back(d);
  }

  std::vector<int> size_;
  std::shared_ptr<Mat> w_;
  std::shared_ptr<Mat> dw_;
};

class Data
{
 public:
  Data()
  {
  }

  std::vector<std::string> sentences_;
  std::set<char> vocab_;
  std::map<char, int> letter_to_index_;
  std::map<int, char> index_to_letter_;
};

class Object
{
 public:
  Object()
  {
  }
  virtual ~Object()
  {
  }

  virtual std::shared_ptr<MatWdw> Forward() = 0;
  virtual void Backward() = 0;
  virtual void ClearDw() = 0;
};

class Graph
{
 public:
  Graph()
  {
  }

  void Process(std::shared_ptr<Object> obj)
  {
    forward_.emplace_back(obj);
    backward_.emplace_back(obj);
  }

  void Forward(bool need_clear = true)
  {
    for (int i = 0; i < forward_.size(); ++i)
    {
      forward_[i]->Forward();
    }
    if (need_clear) forward_.clear();
  }

  void Backward(bool need_clear = true)
  {
    for (int i = backward_.size() - 1; i >= 0; --i)
    {
      backward_[i]->Backward();
    }
    if (need_clear) backward_.clear();
  }

  void ClearDw()
  {
    for (int i = backward_.size() - 1; i >= 0; --i)
    {
      backward_[i]->ClearDw();
    }
  }

 private:
  std::vector<std::shared_ptr<Object>> backward_;
  std::vector<std::shared_ptr<Object>> forward_;
};

class Model
{
 public:
  Model()
  {
  }
  virtual ~Model()
  {
  }

  virtual void Create(std::shared_ptr<Graph> &graph, int idx) = 0;

  virtual void ClearPrevState() = 0;

  std::shared_ptr<MatWdw> input_, output_;

  std::vector<std::shared_ptr<MatWdw>> params_;
  std::vector<std::shared_ptr<MatWdw>> params_prev_;
};

inline float Random01()
{
  return ((float)rand() / RAND_MAX);
}

inline float Randf(float l, float r)
{
  return Random01() * (r - l) + l;
}

inline int Randi(int l, int r)
{
  return floor(Randf(l, r));
}

std::shared_ptr<MatWdw> RandMat(int n, int d, float l, float r);

std::shared_ptr<MatWdw> RandMatGauss(int n, int d, float mean, float stddev);

std::shared_ptr<Mat> Softmax(std::shared_ptr<Mat> &mat);

int MaxIdx(const std::vector<float> &w);

int SampleIdx(std::vector<float> &w);

void Trim(std::string *str);

#endif
