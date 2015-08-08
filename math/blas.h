#ifndef MATH_BLAS_H
#define MATH_BLAS_H

#include "common.h"

class MathBlas : public Math
{
 public:
  MathBlas()
  {
  }
  ~MathBlas()
  {
  }

  void Init();
  void Deinit();

  int Add(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
          std::shared_ptr<Mat> &out);
  int ElmtMul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
              std::shared_ptr<Mat> &out);
  int Mul(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
          std::shared_ptr<Mat> &out);

  int AddDeriv(std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
               std::shared_ptr<Mat> &out);
  int ElmtMulDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                   std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
                   std::shared_ptr<Mat> &out);
  int MulDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
               std::shared_ptr<Mat> &mat1d, std::shared_ptr<Mat> &mat2d,
               std::shared_ptr<Mat> &out);

  int Relu(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out);
  int Sigm(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out);
  int Tanh(std::shared_ptr<Mat> &mat, std::shared_ptr<Mat> &out);

  int ReluDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                std::shared_ptr<Mat> &out);
  int SigmDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                std::shared_ptr<Mat> &out);
  int TanhDeriv(std::shared_ptr<Mat> &mat1, std::shared_ptr<Mat> &mat2,
                std::shared_ptr<Mat> &out);
};

#endif
