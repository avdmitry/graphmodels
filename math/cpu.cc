#include "cpu.h"

#include <stdio.h>
#include <math.h>
#include <float.h>

using std::string;
using std::vector;
using std::shared_ptr;
using std::swap;
using std::max;

void SgemmCpu(bool rowMajor, bool TransA, bool TransB, int M, int N, int K,
              float alpha, float *A, int lda, float *B, int ldb, float beta,
              float *C, int ldc)
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
    swap(dia, dla);
    swap(djb, dlb);
    swap(dic, djc);
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

int MathCpu::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                 shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    out->data_[i] = mat1->data_[i] + mat2->data_[i];
  }
  return 0;
}

int MathCpu::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                     shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    out->data_[i] = mat1->data_[i] * mat2->data_[i];
  }
  return 0;
}

int MathCpu::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                 shared_ptr<Mat> &out)
{
  int m = mat1->size_[0];
  int k2 = mat1->size_[1] * mat1->size_[2] * mat1->size_[3];
  int k = mat2->size_[0];
  int n = mat2->size_[1] * mat2->size_[2] * mat2->size_[3];
  int m2 = out->size_[0];
  int n2 = out->size_[1] * out->size_[2] * out->size_[3];
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

int MathCpu::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                      shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1d->data_.size(); i++)
  {
    float dw = out->data_[i];
    mat1d->data_[i] += dw;
    mat2d->data_[i] += dw;
  }

  return 0;
}

int MathCpu::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                          shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                          shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    float dw = out->data_[i];
    mat1d->data_[i] += mat2->data_[i] * dw;
    mat2d->data_[i] += mat1->data_[i] * dw;
  }

  return 0;
}

int MathCpu::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                      shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                      shared_ptr<Mat> &out)
{
  int mat1_size1 = mat1->size_[1];
  int mat2_size1 = mat2->size_[1];
  for (int i = 0; i < mat1->size_[0]; i++)
  {  // loop over rows of m1
    for (int j = 0; j < mat2_size1; j++)
    {  // loop over cols of m2
      for (int k = 0; k < mat1_size1; k++)
      {
        float dw = out->data_[mat2_size1 * i + j];
        int idx1 = mat1_size1 * i + k;
        int idx2 = mat2_size1 * k + j;
        mat1d->data_[idx1] += mat2->data_[idx2] * dw;
        mat2d->data_[idx2] += mat1->data_[idx1] * dw;
      }
    }
  }
  return 0;
}

int MathCpu::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  for (int i = 0; i < in_w->data_.size(); i++)
  {
    out_w->data_[i] = max(0.0f, in_w->data_[i]);
  }
  return 0;
}

int MathCpu::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  for (int i = 0; i < in_w->data_.size(); i++)
  {
    out_w->data_[i] = 1.0 / (1 + exp(-in_w->data_[i]));
  }
  return 0;
}

int MathCpu::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w)
{
  for (int i = 0; i < in_w->data_.size(); i++)
  {
    out_w->data_[i] = tanh(in_w->data_[i]);
  }
  return 0;
}

int MathCpu::ReluDeriv(shared_ptr<Mat> &in_dw, shared_ptr<Mat> &out_w,
                       shared_ptr<Mat> &out_dw)
{
  for (size_t i = 0; i < out_dw->data_.size(); i++)
  {
    if (out_w->data_[i] > 0)
    {
      in_dw->data_[i] += out_dw->data_[i];
    }
  }
  return 0;
}

int MathCpu::SigmDeriv(shared_ptr<Mat> &in_dw, shared_ptr<Mat> &out_w,
                       shared_ptr<Mat> &out_dw)
{
  for (size_t i = 0; i < out_dw->data_.size(); i++)
  {
    float mwi = out_w->data_[i];
    in_dw->data_[i] += mwi * (1.0 - mwi) * out_dw->data_[i];
  }
  return 0;
}

int MathCpu::TanhDeriv(shared_ptr<Mat> &in_dw, shared_ptr<Mat> &out_w,
                       shared_ptr<Mat> &out_dw)
{
  for (size_t i = 0; i < out_dw->data_.size(); i++)
  {
    float mwi = out_w->data_[i];
    in_dw->data_[i] += (1.0 - mwi * mwi) * out_dw->data_[i];
  }
  return 0;
}

shared_ptr<Mat> MathCpu::Softmax(shared_ptr<Mat> &mat)
{
  shared_ptr<Mat> out(
      new Mat(mat->size_[0], mat->size_[1], mat->size_[2], mat->size_[3]));
  float maxval = mat->data_[0];
  for (int i = 0; i < mat->data_.size(); i++)
  {
    if (mat->data_[i] > maxval)
    {
      maxval = mat->data_[i];
    }
  }

  float sum = 0.0;
  for (int i = 0; i < out->data_.size(); i++)
  {
    out->data_[i] = exp(mat->data_[i] - maxval);
    sum += out->data_[i];
  }
  for (int i = 0; i < out->data_.size(); i++)
  {
    out->data_[i] /= sum;
  }

  return out;
}

int MathCpu::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                  shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                  ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int num_input = conv_params.num_input_channels;
  int num_filters = conv_params.num_output_channels;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  float *in_w_data = &in_w->data_[0];
  float *filters_w_data = &filters_w->data_[0];
  float *biases_w_data = &biases_w->data_[0];
  float *out_w_data = &out_w->data_[0];

  int out_width = (in_width + 2 * padding_x - filter_width) / stride_x + 1;
  int out_height = (in_height + 2 * padding_y - filter_height) / stride_y + 1;

  for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
  {
    int filter_start_x = -padding_x;
    for (int out_x = 0; out_x < out_width; filter_start_x += stride_x, ++out_x)
    {
      int filter_start_y = -padding_y;
      for (int out_y = 0; out_y < out_height;
           filter_start_y += stride_y, ++out_y)
      {
        float res = 0.0;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x)
        {
          int in_x = filter_start_x + filter_x;
          if (in_x < 0 || in_x >= in_width)
          {
            continue;
          }

          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            int in_y = filter_start_y + filter_y;
            if (in_y < 0 || in_y >= in_height)
            {
              continue;
            }

            for (int in_channel = 0; in_channel < num_input; ++in_channel)
            {
              int filter_idx =
                  ((filter_channel * num_input + in_channel) * filter_height +
                   filter_y) *
                      filter_width +
                  filter_x;
              int in_idx = (in_channel * in_height + in_y) * in_width + in_x;
              res += in_w_data[in_idx] * filters_w_data[filter_idx];
            }
          }
        }
        res += biases_w_data[filter_channel];
        out_w_data[(filter_channel * out_height + out_y) * out_width + out_x] =
            res;
      }
    }
  }

  return 0;
}

int MathCpu::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                       shared_ptr<Mat> &filters_w, shared_ptr<Mat> &filters_dw,
                       shared_ptr<Mat> &biases_dw, shared_ptr<Mat> &out_w,
                       shared_ptr<Mat> &out_dw, ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int num_input = conv_params.num_input_channels;
  int num_filters = conv_params.num_output_channels;
  int in_width = in_dw->size_[0];
  int in_height = in_dw->size_[1];
  float *in_w_data = &in_w->data_[0];
  float *in_dw_data = &in_dw->data_[0];
  float *filters_w_data = &filters_w->data_[0];
  float *filters_dw_data = &filters_dw->data_[0];
  float *biases_dw_data = &biases_dw->data_[0];
  float *out_dw_data = &out_dw->data_[0];

  int out_width = (in_width + 2 * padding_x - filter_width) / stride_x + 1;
  int out_height = (in_height + 2 * padding_y - filter_height) / stride_y + 1;

  for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
  {
    int filter_start_x = -padding_x;
    for (int out_x = 0; out_x < out_width; filter_start_x += stride_x, ++out_x)
    {
      int filter_start_y = -padding_y;
      for (int out_y = 0; out_y < out_height;
           filter_start_y += stride_y, ++out_y)
      {
        // grad
        int dw_idx = (filter_channel * out_height + out_y) * out_width + out_x;
        float dw = out_dw_data[dw_idx];

        for (int filter_x = 0; filter_x < filter_width; ++filter_x)
        {
          int in_x = filter_start_x + filter_x;
          if (in_x < 0 || in_x >= in_width)
          {
            continue;
          }

          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            int in_y = filter_start_y + filter_y;
            if (in_y < 0 || in_y >= in_height)
            {
              continue;
            }

            for (int in_channel = 0; in_channel < num_input; ++in_channel)
            {
              int filter_idx =
                  ((filter_channel * num_input + in_channel) * filter_height +
                   filter_y) *
                      filter_width +
                  filter_x;
              int in_idx = (in_channel * in_height + in_y) * in_width + in_x;

              in_dw_data[in_idx] += filters_w_data[filter_idx] * dw;
              filters_dw_data[filter_idx] += in_w_data[in_idx] * dw;
            }
          }
        }
        biases_dw_data[filter_channel] += dw;
      }
    }
  }

  return 0;
}

int MathCpu::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                     ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int num_filters = in_w->size_[2];
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  float *in_w_data = &in_w->data_[0];
  float *out_w_data = &out_w->data_[0];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
  {
    int filter_start_x = -padding_x;
    for (int out_x = 0; out_x < out_width; filter_start_x += stride_x, ++out_x)
    {
      int filter_start_y = -padding_y;
      for (int out_y = 0; out_y < out_height;
           filter_start_y += stride_y, ++out_y)
      {
        float res = -FLT_MAX;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x)
        {
          int in_x = filter_start_x + filter_x;
          if (in_x < 0 || in_x >= in_width)
          {
            continue;
          }

          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            int in_y = filter_start_y + filter_y;
            if (in_y < 0 || in_y >= in_height)
            {
              continue;
            }

            int idx = (filter_channel * in_height + in_y) * in_width + in_x;
            float curr = in_w_data[idx];
            if (curr > res)
            {
              res = curr;
            }
          }
        }

        out_w_data[(filter_channel * out_height + out_y) * out_width + out_x] =
            res;
      }
    }
  }

  return 0;
}

int MathCpu::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &in_dw,
                          shared_ptr<Mat> &out_w, shared_ptr<Mat> &out_dw,
                          ConvParams &conv_params)
{
  int padding_x = conv_params.padding_x;
  int padding_y = conv_params.padding_y;
  int stride_x = conv_params.stride_x;
  int stride_y = conv_params.stride_y;
  int filter_width = conv_params.filter_width;
  int filter_height = conv_params.filter_height;
  int num_filters = in_w->size_[2];
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  float *in_w_data = &in_w->data_[0];
  float *in_dw_data = &in_dw->data_[0];
  float *out_dw_data = &out_dw->data_[0];

  int out_width = (in_width + padding_x * 2 - filter_width) / stride_x + 1;
  int out_height = (in_height + padding_y * 2 - filter_height) / stride_y + 1;

  for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
  {
    int filter_start_x = -padding_x;
    for (int out_x = 0; out_x < out_width; filter_start_x += stride_x, ++out_x)
    {
      int filter_start_y = -padding_y;
      for (int out_y = 0; out_y < out_height;
           filter_start_y += stride_y, ++out_y)
      {
        float res = -FLT_MAX;
        int res_idx = 0;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x)
        {
          int in_x = filter_start_x + filter_x;
          if (in_x < 0 || in_x >= in_width)
          {
            continue;
          }

          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            int in_y = filter_start_y + filter_y;
            if (in_y < 0 || in_y >= in_height)
            {
              continue;
            }

            int idx = (filter_channel * in_height + in_y) * in_width + in_x;
            float curr = in_w_data[idx];
            if (curr > res)
            {
              res = curr;
              res_idx = idx;
            }
          }
        }

        in_dw_data[res_idx] +=
            out_dw_data[(filter_channel * out_height + out_y) * out_width +
                        out_x];
      }
    }
  }

  return 0;
}
