#ifndef MATH_BLAS_H
#define MATH_BLAS_H

#include "common.h"

class MathBlas : public Math
{
 public:
  MathBlas()
  {
  }
  virtual ~MathBlas()
  {
  }

  virtual void Init();
  virtual void Deinit();

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

  virtual int Relu(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out);
  virtual int Sigm(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out);
  virtual int Tanh(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out);

  virtual int ReluDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                        std::shared_ptr<Mat> &out);
  virtual int SigmDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                        std::shared_ptr<Mat> &out);
  virtual int TanhDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                        std::shared_ptr<Mat> &out);

  virtual std::shared_ptr<Mat> Softmax(std::shared_ptr<Mat> &mat);
};

#endif
