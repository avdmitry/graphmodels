#ifndef MATH_CUDNN_H
#define MATH_CUDNN_H

#include "common.h"

class MathCudnn : public Math
{
 public:
  MathCudnn(int gpu_id) : gpu_id_(gpu_id)
  {
  }
  virtual ~MathCudnn()
  {
    Deinit();
  }

  virtual void Init();
  virtual void Deinit();

  virtual void MemoryAlloc(std::shared_ptr<Mat> &mat);
  virtual void MemoryFree(float *ptr);
  virtual void MemoryClear(std::shared_ptr<Mat> &mat);
  virtual void CopyToDevice(std::shared_ptr<Mat> &mat);
  virtual void CopyToHost(std::shared_ptr<Mat> &mat);

  virtual void Sync();

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

  virtual void BatchNormSetUp(std::shared_ptr<Mat> &in_w, Params &params);
  virtual void BatchNorm(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &scale,
                         std::shared_ptr<Mat> &bias, std::shared_ptr<Mat> &mean,
                         std::shared_ptr<Mat> &variance,
                         std::shared_ptr<Mat> &out_w, Params &params,
                         bool train);
  virtual void BatchNormDeriv(std::shared_ptr<Mat> &in_w,
                              std::shared_ptr<Mat> &scale,
                              std::shared_ptr<Mat> &bias,
                              std::shared_ptr<Mat> &mean,
                              std::shared_ptr<Mat> &variance,
                              std::shared_ptr<Mat> &out_w, Params &params);

  virtual void ActivSetUp(std::shared_ptr<Mat> &in_w, Params &params);
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

  virtual void ConvSetUp(std::shared_ptr<Mat> &in_w, Params &params);
  virtual void Conv(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &filters_w,
                    std::shared_ptr<Mat> &biases_w, std::shared_ptr<Mat> &out_w,
                    Params &params);
  virtual void ConvDeriv(std::shared_ptr<Mat> &in_w,
                         std::shared_ptr<Mat> &filters_w,
                         std::shared_ptr<Mat> &biases_w,
                         std::shared_ptr<Mat> &out_w, Params &params);

  virtual void PoolSetUp(std::shared_ptr<Mat> &in_w, PoolType &type,
                         Params &params);
  virtual void MaxPool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                       Params &params);
  virtual void MaxPoolDeriv(std::shared_ptr<Mat> &in_w,
                            std::shared_ptr<Mat> &out_w, Params &params);
  virtual void AvePool(std::shared_ptr<Mat> &in_w, std::shared_ptr<Mat> &out_w,
                       Params &params);
  virtual void AvePoolDeriv(std::shared_ptr<Mat> &in_w,
                            std::shared_ptr<Mat> &out_w, Params &params);

  virtual void SGD(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &mat_prev,
                   float learning_rate, int batch_size, float decay_rate);

  virtual void Rmsprop(std::shared_ptr<Mat> &mat,
                       std::shared_ptr<Mat> &mat_prev, float learning_rate,
                       int batch_size);

 private:
  int gpu_id_;
};

#endif
