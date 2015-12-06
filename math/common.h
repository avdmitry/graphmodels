#ifndef MATH_COMMON_H
#define MATH_COMMON_H

#include <vector>
#include <memory>

#include "mat.h"

enum PoolType
{
  MAX = 0,
  AVE = 1
};

class Context
{
};

class Params
{
 public:
  int num_input;
  int num_output;
  int filter_width;
  int filter_height;
  int stride_x;
  int stride_y;
  int padding_x;
  int padding_y;
  int out_width;
  int out_height;
  std::shared_ptr<Context> context;
};

class Math
{
 public:
  Math()
  {
  }
  virtual ~Math()
  {
  }

  virtual void Init() = 0;
  virtual void Deinit() = 0;

  virtual void MemoryAlloc(std::shared_ptr<Mat> &mat) = 0;
  virtual void MemoryFree(float *ptr) = 0;
  virtual void MemoryClear(std::shared_ptr<Mat> &mat) = 0;
  virtual void CopyToDevice(std::shared_ptr<Mat> &mat) = 0;
  virtual void CopyToHost(std::shared_ptr<Mat> &mat) = 0;

  virtual void Sync()
  {
  }

  virtual void Add(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                   std::shared_ptr<Mat> &out) = 0;
  virtual void ElmtMul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                       std::shared_ptr<Mat> &out) = 0;
  virtual void Mul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                   std::shared_ptr<Mat> &out) = 0;

  virtual void AddDeriv(std::shared_ptr<Mat> &mat1d,
                        std::shared_ptr<Mat> &mat2d,
                        std::shared_ptr<Mat> &out) = 0;
  virtual void ElmtMulDeriv(std::shared_ptr<Mat> &mat1,
                            std::shared_ptr<Mat> &mat2,
                            std::shared_ptr<Mat> &mat1d,
                            std::shared_ptr<Mat> &mat2d,
                            std::shared_ptr<Mat> &out) = 0;
  virtual void MulDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                        std::shared_ptr<Mat> &mat1d,
                        std::shared_ptr<Mat> &mat2d,
                        std::shared_ptr<Mat> &out) = 0;

  virtual void ActivSetUp(std::shared_ptr<Mat> &in_w, Params &params)
  {
  }
  virtual void Relu(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                    Params &params) = 0;
  virtual void Sigm(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                    Params &params) = 0;
  virtual void Tanh(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                    Params &params) = 0;

  virtual void ReluDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &out_w, Params &params) = 0;
  virtual void SigmDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &out_w, Params &params) = 0;
  virtual void TanhDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &out_w, Params &params) = 0;

  virtual float Softmax(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out,
                        std::shared_ptr<Mat> &labels) = 0;

  virtual void Fc(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                  std::shared_ptr<Mat> &biases, std::shared_ptr<Mat> &out) = 0;
  virtual void FcDeriv(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                       std::shared_ptr<Mat> &biases,
                       std::shared_ptr<Mat> &out) = 0;

  virtual void ConvSetUp(std::shared_ptr<Mat> &in_w, Params &params)
  {
  }
  virtual void Conv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &filters_w,
                    std::shared_ptr<Mat> &biases_w, std::shared_ptr<Mat> &out_w,
                    Params &params) = 0;
  virtual void ConvDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &filters_w,
                         std::shared_ptr<Mat> &biases_w,
                         std::shared_ptr<Mat> &out_w, Params &params) = 0;

  virtual void PoolSetUp(std::shared_ptr<Mat> &in_w, PoolType &type,
                         Params &params)
  {
  }
  virtual void MaxPool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                       Params &params) = 0;
  virtual void MaxPoolDeriv(std::shared_ptr<Mat> &in_w,
                            std::shared_ptr<Mat> &out_w, Params &params) = 0;
  virtual void AvePool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                       Params &params) = 0;
  virtual void AvePoolDeriv(std::shared_ptr<Mat> &in_w,
                            std::shared_ptr<Mat> &out_w, Params &params) = 0;

  virtual void SGD(std::shared_ptr<Mat> &mat, float learning_rate,
                   int batch_size) = 0;

  virtual void Rmsprop(std::shared_ptr<Mat> &mat,
                       std::shared_ptr<Mat> &mat_prev, float learning_rate,
                       int batch_size) = 0;
};

#include "math_cpu.h"
#include "math_blas.h"
#include "math_cuda.h"
#include "math_cudnn.h"

extern std::shared_ptr<Math> math;

#endif
