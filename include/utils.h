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

  virtual std::shared_ptr<Mat> Forward() = 0;
  virtual void Backward() = 0;
  virtual void SetBatchSize(int new_size) = 0;
  virtual void ClearDw() = 0;
  virtual void GetParams(std::vector<std::shared_ptr<Mat>> &params) = 0;
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
    if (need_clear)
    {
      forward_.clear();
    }
  }

  void Backward(bool need_clear = true)
  {
    for (int i = backward_.size() - 1; i >= 0; --i)
    {
      backward_[i]->Backward();
    }
    if (need_clear)
    {
      backward_.clear();
    }
  }

  void SetBatchSize(int new_size)
  {
    for (int i = 0; i < forward_.size(); ++i)
    {
      forward_[i]->SetBatchSize(new_size);
    }
  }

  void ClearDw()
  {
    for (int i = backward_.size() - 1; i >= 0; --i)
    {
      backward_[i]->ClearDw();
    }
  }

  void GetParams(std::vector<std::shared_ptr<Mat>> &params)
  {
    for (int i = backward_.size() - 1; i >= 0; --i)
    {
      backward_[i]->GetParams(params);
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

  void Forward()
  {
    graph_->Forward(false);
  }

  void Backward()
  {
    graph_->Backward(false);
  }

  void SetBatchSize(int new_size)
  {
    graph_->SetBatchSize(new_size);
  }

  virtual void Create(int idx) = 0;

  virtual void ClearPrevState() = 0;

  std::shared_ptr<Mat> input_, output_;

  std::shared_ptr<Graph> graph_;

  std::vector<std::shared_ptr<Mat>> params_;
  std::vector<std::shared_ptr<Mat>> params_prev_;
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

std::shared_ptr<Mat> RandMat(int n, int d, float l, float r);

std::shared_ptr<Mat> RandMatGauss(int n, int d, int m, int f, float mean,
                                  float stddev);

int MaxIdx(const std::shared_ptr<Mat> &mat);

int SampleIdx(std::vector<float> &w);

void Trim(std::string *str);

float SoftmaxLoss(std::shared_ptr<Model> &net, std::vector<int> &idx_target);

#endif
