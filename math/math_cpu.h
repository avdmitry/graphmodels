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

  virtual void MemoryAlloc(std::shared_ptr<Mat> &mat)
  {
  }
  virtual void MemoryFree(float *ptr)
  {
  }
  virtual void MemoryClear(std::shared_ptr<Mat> &mat)
  {
  }
  virtual void CopyToDevice(std::shared_ptr<Mat> &mat)
  {
  }
  virtual void CopyToHost(std::shared_ptr<Mat> &mat)
  {
  }

  virtual void Add(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                   std::shared_ptr<Mat> &out);
  virtual void ElmtMul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                       std::shared_ptr<Mat> &out);
  virtual void Mul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                   std::shared_ptr<Mat> &out);

  virtual void AddDeriv(std::shared_ptr<Mat> &mat1d,
                        std::shared_ptr<Mat> &mat2d, std::shared_ptr<Mat> &out);
  virtual void ElmtMulDeriv(std::shared_ptr<Mat> &mat1,
                            std::shared_ptr<Mat> &mat2,
                            std::shared_ptr<Mat> &mat1d,
                            std::shared_ptr<Mat> &mat2d,
                            std::shared_ptr<Mat> &out);
  virtual void MulDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                        std::shared_ptr<Mat> &mat1d,
                        std::shared_ptr<Mat> &mat2d, std::shared_ptr<Mat> &out);

  virtual void Relu(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                    Params &params);
  virtual void Sigm(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                    Params &params);
  virtual void Tanh(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                    Params &params);

  virtual void ReluDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &out_w, Params &params);
  virtual void SigmDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &out_w, Params &params);
  virtual void TanhDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &out_w, Params &params);

  virtual float Softmax(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out,
                        std::shared_ptr<Mat> &labels);

  virtual void Fc(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                  std::shared_ptr<Mat> &biases, std::shared_ptr<Mat> &out);
  virtual void FcDeriv(std::shared_ptr<Mat> &in, std::shared_ptr<Mat> &filters,
                       std::shared_ptr<Mat> &biases, std::shared_ptr<Mat> &out);

  virtual void Conv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &filters_w,
                    std::shared_ptr<Mat> &biases_w, std::shared_ptr<Mat> &out_w,
                    Params &params);
  virtual void ConvDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &filters_w,
                         std::shared_ptr<Mat> &biases_w,
                         std::shared_ptr<Mat> &out_w, Params &params);

  virtual void MaxPool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                       Params &params);
  virtual void MaxPoolDeriv(std::shared_ptr<Mat> &in_w,
                            std::shared_ptr<Mat> &out_w, Params &params);
  virtual void AvePool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                       Params &params);
  virtual void AvePoolDeriv(std::shared_ptr<Mat> &in_w,
                            std::shared_ptr<Mat> &out_w, Params &params);

  virtual void SGD(std::shared_ptr<Mat> &mat, float learning_rate,
                   int batch_size);

  virtual void Rmsprop(std::shared_ptr<Mat> &mat,
                       std::shared_ptr<Mat> &mat_prev, float learning_rate,
                       int batch_size);
};

#endif
