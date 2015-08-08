#ifndef MATH_CPU_H
#define MATH_CPU_H

#include "common.h"

class MathCpu : public Math
{
 public:
  MathCpu()
  {
  }
  ~MathCpu()
  {
  }

  void Init();
  void Deinit();

  int Add(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
          std::shared_ptr<Mat>& out);
  int ElmtMul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
              std::shared_ptr<Mat>& out);
  int Mul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
          std::shared_ptr<Mat>& out);

  int AddDeriv(std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
               std::shared_ptr<Mat>& out);
  int ElmtMulDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                   std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                   std::shared_ptr<Mat>& out);
  int MulDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
               std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
               std::shared_ptr<Mat>& out);

  int Relu(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out);
  int Sigm(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out);
  int Tanh(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out);

  int ReluDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                std::shared_ptr<Mat>& out);
  int SigmDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                std::shared_ptr<Mat>& out);
  int TanhDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                std::shared_ptr<Mat>& out);
};

void SgemmCpu(bool rowMajor, bool TransA, bool TransB, int M, int N, int K,
              float alpha, float* A, int lda, float* B, int ldb, float beta,
              float* C, int ldc);

#endif
