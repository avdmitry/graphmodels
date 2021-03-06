#ifdef BUILD_BLAS
#include "math_blas.h"

#include <stdio.h>
#include <math.h>

#include <cblas.h>

#include "math_cpu.h"  // SgemmCpu, default implementation

using std::string;
using std::vector;
using std::shared_ptr;

static shared_ptr<MathCpu> math_cpu(new MathCpu);

void SgemmBlas(bool rowMajor, bool TransA, bool TransB, int M, int N, int K,
               float alpha, float *A, int lda, float *B, int ldb, float beta,
               float *C, int ldc)
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

void MathBlas::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                   shared_ptr<Mat> &out)
{
  float alpha = 1.0f;
  cblas_scopy(out->data_.size(), &mat1->data_[0], 1, &out->data_[0], 1);
  cblas_saxpy(out->data_.size(), alpha, &mat2->data_[0], 1, &out->data_[0], 1);
}

void MathBlas::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                       shared_ptr<Mat> &out)
{
  int m = mat1->size_[0] * mat1->size_[1];

  float alpha = 1.0f;
  float beta = 0.0f;
  cblas_sgbmv(CblasRowMajor, CblasNoTrans, m, m, 0, 0, alpha, &mat1->data_[0],
              1, &mat2->data_[0], 1, beta, &out->data_[0], 1);
}

void MathBlas::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                   shared_ptr<Mat> &out)
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
    return;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  SgemmBlas(true, false, false, m, n, k, alpha, &mat1->data_[0], mat1->size_[1],
            &mat2->data_[0], mat2->size_[1], beta, &out->data_[0],
            out->size_[1]);
}

void MathBlas::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                        shared_ptr<Mat> &out)
{
  math_cpu->AddDeriv(mat1d, mat2d, out);
}

void MathBlas::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                            shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                            shared_ptr<Mat> &out)
{
  math_cpu->ElmtMulDeriv(mat1, mat2, mat1d, mat2d, out);
}

void MathBlas::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                        shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                        shared_ptr<Mat> &out)
{
  math_cpu->MulDeriv(mat1, mat2, mat1d, mat2d, out);
}

void MathBlas::BatchNorm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &scale,
                         shared_ptr<Mat> &bias, shared_ptr<Mat> &mean,
                         shared_ptr<Mat> &variance, shared_ptr<Mat> &out_w,
                         Params &params, bool train)
{
  math_cpu->BatchNorm(in_w, scale, bias, mean, variance, out_w, params, train);
}

void MathBlas::BatchNormDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &scale,
                              shared_ptr<Mat> &bias, shared_ptr<Mat> &mean,
                              shared_ptr<Mat> &variance, shared_ptr<Mat> &out_w,
                              Params &params)
{
  math_cpu->BatchNormDeriv(in_w, scale, bias, mean, variance, out_w, params);
}

// Activation functions here currently the same as for Cpu.
void MathBlas::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  math_cpu->Relu(in_w, out_w, params);
}

void MathBlas::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  math_cpu->Sigm(in_w, out_w, params);
}

void MathBlas::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  math_cpu->Tanh(in_w, out_w, params);
}

void MathBlas::ReluDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  math_cpu->ReluDeriv(in_w, out_w, params);
}

void MathBlas::SigmDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  math_cpu->SigmDeriv(in_w, out_w, params);
}

void MathBlas::TanhDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  math_cpu->TanhDeriv(in_w, out_w, params);
}

float MathBlas::Softmax(shared_ptr<Mat> &mat, shared_ptr<Mat> &out,
                        shared_ptr<Mat> &labels)
{
  return math_cpu->Softmax(mat, out, labels);
}

void MathBlas::Fc(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                  shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  math_cpu->Fc(in, filters, biases, out);
}

void MathBlas::FcDeriv(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                       shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  math_cpu->FcDeriv(in, filters, biases, out);
}

void MathBlas::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                    shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                    Params &params)
{
  math_cpu->Conv(in_w, filters_w, biases_w, out_w, params);
}

void MathBlas::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                         shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                         Params &params)
{
  math_cpu->ConvDeriv(in_w, filters_w, biases_w, out_w, params);
}

void MathBlas::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                       Params &params)
{
  math_cpu->MaxPool(in_w, out_w, params);
}

void MathBlas::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                            Params &params)
{
  math_cpu->MaxPoolDeriv(in_w, out_w, params);
}

void MathBlas::AvePool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                       Params &params)
{
  math_cpu->AvePool(in_w, out_w, params);
}

void MathBlas::AvePoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                            Params &params)
{
  math_cpu->AvePoolDeriv(in_w, out_w, params);
}

void MathBlas::SGD(shared_ptr<Mat> &mat, shared_ptr<Mat> &mat_prev,
                   float learning_rate, int batch_size, float decay_rate)
{
  math_cpu->SGD(mat, mat_prev, learning_rate, batch_size, decay_rate);
}

void MathBlas::Rmsprop(shared_ptr<Mat> &mat, shared_ptr<Mat> &mat_prev,
                       float learning_rate, int batch_size)
{
  math_cpu->Rmsprop(mat, mat_prev, learning_rate, batch_size);
}

#endif
