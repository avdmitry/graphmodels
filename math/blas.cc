#include "blas.h"

#include <stdio.h>
#include <math.h>

#include <cblas.h>

#include "cpu.h"  // SgemmCpu, default implementation

using namespace std;

static shared_ptr<MathCpu> math_cpu(new MathCpu);

void SgemmBlas(bool rowMajor, bool TransA, bool TransB, int M, int N, int K,
               float alpha, float* A, int lda, float* B, int ldb, float beta,
               float* C, int ldc)
{
  CBLAS_TRANSPOSE ta;
  if (TransA)
  {
    ta = CblasTrans;
  }
  else
  {
    ta = CblasNoTrans;
  }
  CBLAS_TRANSPOSE tb;
  if (TransB)
  {
    tb = CblasTrans;
  }
  else
  {
    tb = CblasNoTrans;
  }
  CBLAS_ORDER order;
  if (rowMajor)
  {
    order = CblasRowMajor;
  }
  else
  {
    order = CblasColMajor;
  }

  cblas_sgemm(order, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void MathBlas::Init()
{
}

void MathBlas::Deinit()
{
}

int MathBlas::Mul(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                  shared_ptr<Mat>& out)
{
  int m = mat1->size_[0];
  int k2 = mat1->size_[1];
  int k = mat2->size_[0];
  int n = mat2->size_[1];
  int m2 = out->size_[0];
  int n2 = out->size_[1];
  if (m != m2 || n != n2 || k != k2)
  {
    printf("%d %d %d %d %d %d\n", m, k2, k, n, m2, n2);
    return -1;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  SgemmBlas(true, false, false, m, n, k, alpha, &mat1->data_[0], mat1->size_[1],
            &mat2->data_[0], mat2->size_[1], beta, &out->data_[0],
            out->size_[1]);
  return 0;
}

int MathBlas::Add(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                  shared_ptr<Mat>& out)
{
  float alpha = 1.0f;
  cblas_scopy(out->data_.size(), &mat1->data_[0], 1, &out->data_[0], 1);
  cblas_saxpy(out->data_.size(), alpha, &mat2->data_[0], 1, &out->data_[0], 1);
  return 0;
}

int MathBlas::ElmtMul(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                      shared_ptr<Mat>& out)
{
  int m = mat1->size_[0] * mat1->size_[1];

  float alpha = 1.0f;
  float beta = 0.0f;
  cblas_sgbmv(CblasRowMajor, CblasNoTrans, m, m, 0, 0, alpha, &mat1->data_[0],
              1, &mat2->data_[0], 1, beta, &out->data_[0], 1);
  return 0;
}

int MathBlas::AddDeriv(shared_ptr<Mat>& mat1d, shared_ptr<Mat>& mat2d,
                       shared_ptr<Mat>& out)
{
  return math_cpu->AddDeriv(mat1d, mat2d, out);
}

int MathBlas::ElmtMulDeriv(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                           shared_ptr<Mat>& mat1d, shared_ptr<Mat>& mat2d,
                           shared_ptr<Mat>& out)
{
  return math_cpu->ElmtMulDeriv(mat1, mat2, mat1d, mat2d, out);
}

int MathBlas::MulDeriv(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                       shared_ptr<Mat>& mat1d, shared_ptr<Mat>& mat2d,
                       shared_ptr<Mat>& out)
{
  return math_cpu->MulDeriv(mat1, mat2, mat1d, mat2d, out);
}

// Activation functions here currently the same as for Cpu.
int MathBlas::Relu(shared_ptr<Mat>& mat, shared_ptr<Mat>& out)
{
  return math_cpu->Relu(mat, out);
}

int MathBlas::Sigm(shared_ptr<Mat>& mat, shared_ptr<Mat>& out)
{
  return math_cpu->Sigm(mat, out);
}

int MathBlas::Tanh(shared_ptr<Mat>& mat, shared_ptr<Mat>& out)
{
  return math_cpu->Tanh(mat, out);
}

int MathBlas::ReluDeriv(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                        shared_ptr<Mat>& out)
{
  return math_cpu->ReluDeriv(mat1, mat2, out);
}

int MathBlas::SigmDeriv(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                        shared_ptr<Mat>& out)
{
  return math_cpu->SigmDeriv(mat1, mat2, out);
}

int MathBlas::TanhDeriv(shared_ptr<Mat>& mat1, shared_ptr<Mat>& mat2,
                        shared_ptr<Mat>& out)
{
  return math_cpu->TanhDeriv(mat1, mat2, out);
}

shared_ptr<Mat> MathBlas::Softmax(std::shared_ptr<Mat>& mat)
{
  return math_cpu->Softmax(mat);
}
