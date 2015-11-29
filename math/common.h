#ifndef MATH_COMMON_H
#define MATH_COMMON_H

#include <vector>
#include <memory>

#include "mat.h"

class ConvParams
{
 public:
  int num_input_channels;
  int num_output_channels;
  int filter_width;
  int filter_height;
  int stride_x;
  int stride_y;
  int padding_x;
  int padding_y;
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

  virtual int FreeMatMemory(float *ptr) = 0;

  virtual int Add(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                  std::shared_ptr<Mat> &out) = 0;
  virtual int ElmtMul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                      std::shared_ptr<Mat> &out) = 0;
  virtual int Mul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                  std::shared_ptr<Mat> &out) = 0;

  virtual int AddDeriv(std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
                       std::shared_ptr<Mat> &out) = 0;
  virtual int ElmtMulDeriv(std::shared_ptr<Mat> &mat1,
                           std::shared_ptr<Mat> &mat2,
                           std::shared_ptr<Mat> &mat1d,
                           std::shared_ptr<Mat> &mat2d,
                           std::shared_ptr<Mat> &out) = 0;
  virtual int MulDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                       std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
                       std::shared_ptr<Mat> &out) = 0;

  virtual int Relu(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w) = 0;
  virtual int Sigm(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w) = 0;
  virtual int Tanh(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w) = 0;

  virtual int ReluDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw) = 0;
  virtual int SigmDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw) = 0;
  virtual int TanhDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw) = 0;

  virtual std::shared_ptr<Mat> Softmax(std::shared_ptr<Mat> &mat) = 0;

  virtual int Fc(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                 std::shared_ptr<Mat> &biases, std::shared_ptr<Mat> &out) = 0;
  virtual int FcDeriv(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                      std::shared_ptr<Mat> &biases,
                      std::shared_ptr<Mat> &out) = 0;

  virtual int Conv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &filters_w,
                   std::shared_ptr<Mat> &biases_w, std::shared_ptr<Mat> &out_w,
                   ConvParams &conv_params) = 0;
  virtual int ConvDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &filters_w,
                        std::shared_ptr<Mat> &filters_dw,
                        std::shared_ptr<Mat> &biases_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw,
                        ConvParams &conv_params) = 0;

  virtual int MaxPool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                      ConvParams &conv_params) = 0;
  virtual int MaxPoolDeriv(std::shared_ptr<Mat> &in_w,
                           std::shared_ptr<Mat> &in_dw,
                           std::shared_ptr<Mat> &out_w,
                           std::shared_ptr<Mat> &out_dw,
                           ConvParams &conv_params) = 0;
  virtual int AvePool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                      ConvParams &conv_params) = 0;
  virtual int AvePoolDeriv(std::shared_ptr<Mat> &in_w,
                           std::shared_ptr<Mat> &in_dw,
                           std::shared_ptr<Mat> &out_w,
                           std::shared_ptr<Mat> &out_dw,
                           ConvParams &conv_params) = 0;
};

#include "cpu.h"
#include "blas.h"
#include "cuda.h"
#include "cudnn.h"

extern std::shared_ptr<Math> math;

#endif
