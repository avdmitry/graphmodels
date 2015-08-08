#include "cpu.h"

#include <stdio.h>
#include <math.h>
#include <float.h>

void SgemmCpu(bool rowMajor, bool TransA, bool TransB, int M, int N, int K,
              float alpha, float* A, int lda, float* B, int ldb, float beta,
              float* C, int ldc)
{
  int dia, dla;
  if (TransA)
  {
    dia = 1;
    dla = lda;
  }
  else
  {
    dia = lda;
    dla = 1;
  }
  int djb, dlb;
  if (TransB)
  {
    djb = ldb;
    dlb = 1;
  }
  else
  {
    djb = 1;
    dlb = ldb;
  }
  int dic = 1, djc = ldc;

  if (!rowMajor)
  {
    std::swap(dia, dla);
    std::swap(djb, dlb);
    std::swap(dic, djc);
  }

  for (int i = 0, ia = 0, ic = 0; i < M; ++i, ia += dia, ic += djc)
  {
    for (int j = 0, jb = 0, jc = 0; j < N; ++j, jb += djb, jc += dic)
    {
      float res = 0;
      for (int l = 0, la = 0, lb = 0; l < K; ++l, la += dla, lb += dlb)
      {
        res += A[la + ia] * B[lb + jb];
      }
      int target_ind = jc + ic;
      C[target_ind] = beta * C[target_ind] + alpha * res;
    }
  }
}

void MathCpu::Init()
{
}

void MathCpu::Deinit()
{
}

int MathCpu::Add(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                 std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    out->data_[i] = mat1->data_[i] + mat2->data_[i];
  }
  return 0;
}

int MathCpu::ElmtMul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                     std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    out->data_[i] = mat1->data_[i] * mat2->data_[i];
  }
  return 0;
}

int MathCpu::Mul(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                 std::shared_ptr<Mat>& out)
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
  SgemmCpu(true, false, false, m, n, k, alpha, &mat1->data_[0], mat1->size_[1],
           &mat2->data_[0], mat2->size_[1], beta, &out->data_[0],
           out->size_[1]);

  return 0;
}

int MathCpu::AddDeriv(std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                      std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat1d->data_.size(); i++)
  {
    float curr = out->data_[i];
    mat1d->data_[i] += curr;
    mat2d->data_[i] += curr;
  }
  return 0;
}

int MathCpu::ElmtMulDeriv(std::shared_ptr<Mat>& mat1,
                          std::shared_ptr<Mat>& mat2,
                          std::shared_ptr<Mat>& mat1d,
                          std::shared_ptr<Mat>& mat2d,
                          std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    float curr = out->data_[i];
    mat1d->data_[i] += mat2->data_[i] * curr;
    mat2d->data_[i] += mat1->data_[i] * curr;
  }
  return 0;
}

int MathCpu::MulDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                      std::shared_ptr<Mat>& mat1d, std::shared_ptr<Mat>& mat2d,
                      std::shared_ptr<Mat>& out)
{
  int mat1_size1 = mat1->size_[1];
  int mat2_size1 = mat2->size_[1];
  for (int i = 0; i < mat1->size_[0]; i++)
  {  // loop over rows of m1
    for (int j = 0; j < mat2_size1; j++)
    {  // loop over cols of m2
      for (int k = 0; k < mat1_size1; k++)
      {  // dot product loop
        float b = out->data_[mat2_size1 * i + j];
        mat1d->data_[mat1_size1 * i + k] += mat2->data_[mat2_size1 * k + j] * b;
        mat2d->data_[mat2_size1 * k + j] += mat1->data_[mat1_size1 * i + k] * b;
      }
    }
  }
  return 0;
}

int MathCpu::Relu(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat->data_.size(); i++)
  {
    out->data_[i] = std::max(0.0f, mat->data_[i]);
  }
  return 0;
}

int MathCpu::Sigm(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat->data_.size(); i++)
  {
    out->data_[i] = 1.0 / (1 + exp(-mat->data_[i]));
  }
  return 0;
}

int MathCpu::Tanh(std::shared_ptr<Mat>& mat, std::shared_ptr<Mat>& out)
{
  for (int i = 0; i < mat->data_.size(); i++)
  {
    out->data_[i] = tanh(mat->data_[i]);
  }
  return 0;
}

int MathCpu::ReluDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                       std::shared_ptr<Mat>& out)
{
  for (size_t i = 0; i < mat1->data_.size(); i++)
  {
    if (mat2->data_[i] > 0)
    {
      out->data_[i] += mat1->data_[i];
    }
  }
  return 0;
}

int MathCpu::SigmDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                       std::shared_ptr<Mat>& out)
{
  for (size_t i = 0; i < mat1->data_.size(); i++)
  {
    float mwi = mat2->data_[i];
    out->data_[i] += mwi * (1.0 - mwi) * mat1->data_[i];
  }
  return 0;
}

int MathCpu::TanhDeriv(std::shared_ptr<Mat>& mat1, std::shared_ptr<Mat>& mat2,
                       std::shared_ptr<Mat>& out)
{
  for (size_t i = 0; i < mat1->data_.size(); i++)
  {
    float mwi = mat2->data_[i];
    out->data_[i] += (1.0 - mwi * mwi) * mat1->data_[i];
  }
  return 0;
}
