#ifndef MATH_COMMON_H
#define MATH_COMMON_H

#include <vector>
#include <memory>

#include "mat.h"

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

  virtual int Add(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                  std::shared_ptr<Mat>& out) = 0;
  virtual int ElmtMul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                      std::shared_ptr<Mat>& out) = 0;
  virtual int Mul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                  std::shared_ptr<Mat>& out) = 0;

  virtual int AddDeriv(std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                       std::shared_ptr<Mat>& out) = 0;
  virtual int ElmtMulDeriv(std::shared_ptr<Mat>& mat1,
                           std::shared_ptr<Mat>& mat2,
                           std::shared_ptr<Mat>& mat1d,
                           std::shared_ptr<Mat>& mat2d,
                           std::shared_ptr<Mat>& out) = 0;
  virtual int MulDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                       std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                       std::shared_ptr<Mat>& out) = 0;

  virtual int Relu(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out) = 0;
  virtual int Sigm(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out) = 0;
  virtual int Tanh(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out) = 0;

  virtual int ReluDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                        std::shared_ptr<Mat>& out) = 0;
  virtual int SigmDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                        std::shared_ptr<Mat>& out) = 0;
  virtual int TanhDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                        std::shared_ptr<Mat>& out) = 0;
};

#include "cpu.h"
#include "blas.h"
#include "cuda.h"

extern std::shared_ptr<Math> math;

#endif
