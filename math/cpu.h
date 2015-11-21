#ifndef MATH_CPU_H
#define MATH_CPU_H

#include "common.h"

class MathCpu : public Math
{
 public:
  MathCpu()
  {
  }
  virtual ~MathCpu()
  {
    Deinit();
  }

  virtual void Init();
  virtual void Deinit();

  virtual int FreeMatMemory(float *ptr) { return 0; }

  virtual int Add(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                  std::shared_ptr<Mat> &out);
  virtual int ElmtMul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                      std::shared_ptr<Mat> &out);
  virtual int Mul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                  std::shared_ptr<Mat> &out);

  virtual int AddDeriv(std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
                       std::shared_ptr<Mat> &out);
  virtual int ElmtMulDeriv(std::shared_ptr<Mat> &mat1,
                           std::shared_ptr<Mat> &mat2,
                           std::shared_ptr<Mat> &mat1d,
                           std::shared_ptr<Mat> &mat2d,
                           std::shared_ptr<Mat> &out);
  virtual int MulDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                       std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
                       std::shared_ptr<Mat> &out);

  virtual int Relu(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w);
  virtual int Sigm(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w);
  virtual int Tanh(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w);

  virtual int ReluDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw);
  virtual int SigmDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw);
  virtual int TanhDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw);

  virtual std::shared_ptr<Mat> Softmax(std::shared_ptr<Mat> &mat);

  virtual int Fc(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                 std::shared_ptr<Mat> &biases, std::shared_ptr<Mat> &out);
  virtual int FcDeriv(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                      std::shared_ptr<Mat> &biases,
                      std::shared_ptr<Mat> &out);

  virtual int Conv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &filters_w,
                   std::shared_ptr<Mat> &biases_w, std::shared_ptr<Mat> &out_w,
                   ConvParams &conv_params);
  virtual int ConvDeriv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &in_dw,
                        std::shared_ptr<Mat> &filters_w,
                        std::shared_ptr<Mat> &filters_dw,
                        std::shared_ptr<Mat> &biases_dw,
                        std::shared_ptr<Mat> &out_w,
                        std::shared_ptr<Mat> &out_dw, ConvParams &conv_params);

  virtual int MaxPool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                      ConvParams &conv_params);
  virtual int MaxPoolDeriv(std::shared_ptr<Mat> &in_w,
                           std::shared_ptr<Mat> &in_dw,
                           std::shared_ptr<Mat> &out_w,
                           std::shared_ptr<Mat> &out_dw,
                           ConvParams &conv_params);
};

#endif
