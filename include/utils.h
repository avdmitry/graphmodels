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

class Operation
{
 public:
  Operation()
  {
  }
  virtual ~Operation()
  {
  }

  virtual void Forward(bool train) = 0;
  virtual void Backward() = 0;
  virtual void SetBatchSize(int new_size) = 0;
  virtual void ClearDw() = 0;
  virtual void GetParams(std::vector<std::shared_ptr<Mat>> &params) = 0;

  std::string GetName()
  {
    return name_;
  }

 protected:
  std::string name_;
};

class Graph
{
 public:
  Graph()
  {
  }

  void Process(std::shared_ptr<Operation> obj)
  {
    forward_.emplace_back(obj);
    backward_.emplace_back(obj);
  }

  void Forward(bool train, bool need_clear)
  {
    for (int i = 0; i < forward_.size(); ++i)
    {
      forward_[i]->Forward(train);
    }
    if (need_clear)
    {
      forward_.clear();
    }
  }

  void Backward(bool need_clear)
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

  void GetParamsForward(std::vector<std::shared_ptr<Mat>> &params)
  {
    for (int i = 0; i < forward_.size(); ++i)
    {
      forward_[i]->GetParams(params);
    }
  }

  void Save(FILE *file)
  {
    for (int i = 0; i < forward_.size(); ++i)
    {
      std::vector<std::shared_ptr<Mat>> params;
      forward_[i]->GetParams(params);
      if (params.size() > 0)
      {
        std::string name = forward_[i]->GetName();
        int8_t name_size = name.length() + 1;
        fwrite((void *)&name_size, sizeof(int8_t), 1, file);
        fwrite((void *)name.c_str(), 1, name_size, file);
        for (std::shared_ptr<Mat> &curr : params)
        {
          math->CopyToHost(curr);
          int32_t param_size = curr->data_.size();
          fwrite((void *)&param_size, sizeof(int32_t), 1, file);
          fwrite((void *)&curr->data_[0], sizeof(float), param_size, file);
        }
      }
    }
  }

  void Load(FILE *file)
  {
    for (int i = 0; i < forward_.size(); ++i)
    {
      std::vector<std::shared_ptr<Mat>> params;
      forward_[i]->GetParams(params);
      if (params.size() > 0)
      {
        int8_t name_size;
        int res;
        res = fread((void *)&name_size, sizeof(int8_t), 1, file);
        std::string name;
        name.resize(name_size);
        res = fread((void *)name.c_str(), 1, name_size, file);
        printf("%s -> %s\n", name.c_str(), forward_[i]->GetName().c_str());
        for (std::shared_ptr<Mat> &curr : params)
        {
          int32_t param_size;
          res = fread((void *)&param_size, sizeof(int32_t), 1, file);
          // printf("%u -> %lu\n", param_size, curr->data_.size());
          res = fread((void *)&curr->data_[0], sizeof(float), param_size, file);
          math->CopyToDevice(curr);
        }
      }
    }
  }

 private:
  std::vector<std::shared_ptr<Operation>> backward_;
  std::vector<std::shared_ptr<Operation>> forward_;
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

  void Forward(bool train)
  {
    graph_->Forward(train, false);
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

std::shared_ptr<std::vector<std::string>> Split(const std::string &s,
                                                char delim);

#endif
