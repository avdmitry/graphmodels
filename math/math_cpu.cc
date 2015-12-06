#include "math_cpu.h"

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

void MathCpu::Add(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                  shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    out->data_[i] = mat1->data_[i] + mat2->data_[i];
  }
}

void MathCpu::ElmtMul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                      shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    out->data_[i] = mat1->data_[i] * mat2->data_[i];
  }
}

void MathCpu::Mul(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
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
  SgemmCpu(true, false, false, m, n, k, alpha, &mat1->data_[0], mat1->size_[1],
           &mat2->data_[0], mat2->size_[1], beta, &out->data_[0],
           out->size_[1]);
}

void MathCpu::AddDeriv(shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                       shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1d->data_.size(); i++)
  {
    float dw = out->data_[i];
    mat1d->data_[i] += dw;
    mat2d->data_[i] += dw;
  }
}

void MathCpu::ElmtMulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                           shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                           shared_ptr<Mat> &out)
{
  for (int i = 0; i < mat1->data_.size(); i++)
  {
    float dw = out->data_[i];
    mat1d->data_[i] += mat2->data_[i] * dw;
    mat2d->data_[i] += mat1->data_[i] * dw;
  }
}

void MathCpu::MulDeriv(shared_ptr<Mat> &mat1, shared_ptr<Mat> &mat2,
                       shared_ptr<Mat> &mat1d, shared_ptr<Mat> &mat2d,
                       shared_ptr<Mat> &out)
{
  int mat1_size1 = mat1->size_[1];
  int mat2_size1 = mat2->size_[1];
  for (int i = 0; i < mat1->size_[0]; ++i)
  {
    for (int j = 0; j < mat2_size1; ++j)
    {
      float dw = out->data_[mat2_size1 * i + j];
      for (int k = 0; k < mat1_size1; ++k)
      {
        int idx1 = mat1_size1 * i + k;
        int idx2 = mat2_size1 * k + j;
        mat1d->data_[idx1] += mat2->data_[idx2] * dw;
        mat2d->data_[idx2] += mat1->data_[idx1] * dw;
      }
    }
  }

  /*int m = mat1d->size_[0];
  int n = mat1d->size_[1];
  int k = mat2->size_[1];
  float alpha = 1.0f;
  float beta = 0.0f;
  SgemmCpu(true, false, true, m, n, k, alpha, &out->data_[0], k,
           &mat2->data_[0], k, beta, &mat1d->data_[0], n);

  m = mat2d->size_[1];
  n = mat1->size_[1];
  k = mat1->size_[0];
  alpha = 1.0f;
  beta = 0.0f;
  SgemmCpu(false, false, true, m, n, k, alpha, &out->data_[0], m,
           &mat1->data_[0], n, beta, &mat2d->data_[0], m);*/
}

void MathCpu::Relu(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                   Params &params)
{
  for (int i = 0; i < in_w->data_.size(); i++)
  {
    out_w->data_[i] = max(0.0f, in_w->data_[i]);
  }
}

void MathCpu::Sigm(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                   Params &params)
{
  for (int i = 0; i < in_w->data_.size(); i++)
  {
    out_w->data_[i] = 1.0 / (1 + exp(-in_w->data_[i]));
  }
}

void MathCpu::Tanh(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                   Params &params)
{
  for (int i = 0; i < in_w->data_.size(); i++)
  {
    out_w->data_[i] = tanh(in_w->data_[i]);
  }
}

void MathCpu::ReluDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                        Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;
  for (size_t i = 0; i < out_dw->data_.size(); i++)
  {
    if (out_w->data_[i] > 0)
    {
      in_dw->data_[i] += out_dw->data_[i];
    }
  }
}

void MathCpu::SigmDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                        Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;
  for (size_t i = 0; i < out_dw->data_.size(); i++)
  {
    float mwi = out_w->data_[i];
    in_dw->data_[i] += mwi * (1.0 - mwi) * out_dw->data_[i];
  }
}

void MathCpu::TanhDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                        Params &params)
{
  shared_ptr<Mat> &in_dw = in_w->dw_;
  shared_ptr<Mat> &out_dw = out_w->dw_;
  for (size_t i = 0; i < out_dw->data_.size(); i++)
  {
    float mwi = out_w->data_[i];
    in_dw->data_[i] += (1.0 - mwi * mwi) * out_dw->data_[i];
  }
}

float MathCpu::Softmax(shared_ptr<Mat> &mat, shared_ptr<Mat> &out,
                       shared_ptr<Mat> &labels)
{
  out = shared_ptr<Mat>(new Mat(mat->size_));
  int num_elements = mat->size_[0] * mat->size_[1] * mat->size_[2];
  for (int batch = 0; batch < mat->size_[3]; ++batch)
  {
    int offset = batch * num_elements;
    float maxval = mat->data_[offset];
    for (int i = 0; i < num_elements; i++)
    {
      if (mat->data_[offset + i] > maxval)
      {
        maxval = mat->data_[offset + i];
      }
    }

    float sum = 0.0;
    for (int i = 0; i < num_elements; i++)
    {
      out->data_[offset + i] = exp(mat->data_[offset + i] - maxval);
      sum += out->data_[offset + i];
    }
    for (int i = 0; i < num_elements; i++)
    {
      out->data_[offset + i] /= sum;
    }
  }

  mat->dw_->data_ = out->data_;
  float loss = 0;
  for (int batch = 0; batch < out->size_[3]; ++batch)
  {
    int idx = batch * num_elements + labels->data_[batch];
    mat->dw_->data_[idx] -= 1;
    loss -= log(out->data_[idx]);
  }

  return loss;
}

void MathCpu::Fc(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                 shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  int num_out = out->size_[2];
  int num_in = filters->size_[0];
  int num_batch = in->size_[3];

  for (int batch = 0; batch < num_batch; ++batch)
  {
    int in_offset = num_in * batch;
    int out_offset = num_out * batch;
    for (int i = 0; i < num_out; ++i)
    {
      float result = biases->data_[i];
      for (int j = 0; j < num_in; ++j)
      {
        // int filters_idx = num_out * j + i;
        int filters_idx = num_in * i + j;
        result += in->data_[in_offset + j] * filters->data_[filters_idx];
      }
      out->data_[out_offset + i] = result;
    }
  }
}

void MathCpu::FcDeriv(shared_ptr<Mat> &in, shared_ptr<Mat> &filters,
                      shared_ptr<Mat> &biases, shared_ptr<Mat> &out)
{
  int num_out = out->size_[2];
  int num_in = filters->size_[0];
  int num_batch = in->size_[3];

  for (int batch = 0; batch < num_batch; ++batch)
  {
    int in_offset = num_in * batch;
    int out_offset = num_out * batch;
    for (int i = 0; i < num_out; ++i)
    {
      float dw = out->dw_->data_[out_offset + i];
      for (int j = 0; j < num_in; ++j)
      {
        // int filters_idx = num_out * j + i;
        int filters_idx = num_in * i + j;
        in->dw_->data_[in_offset + j] += dw * filters->data_[filters_idx];
        filters->dw_->data_[filters_idx] += dw * in->data_[in_offset + j];
      }
      biases->dw_->data_[i] += dw;
    }
  }
}

void MathCpu::Conv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                   shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                   Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int num_input = params.num_input;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];
  int out_width = params.out_width;
  int out_height = params.out_height;

  float *in_w_data = &in_w->data_[0];
  float *filters_w_data = &filters_w->data_[0];
  float *biases_w_data = &biases_w->data_[0];
  float *out_w_data = &out_w->data_[0];

  for (int batch = 0; batch < batch_size; ++batch)
  {
    int in_offset = in_width * in_height * num_input * batch;
    int out_offset = out_width * out_height * num_filters * batch;

    for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
    {
      int filter_start_x = -padding_x;
      for (int out_x = 0; out_x < out_width;
           filter_start_x += stride_x, ++out_x)
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
                int in_idx = in_offset +
                             (in_channel * in_height + in_y) * in_width + in_x;
                res += in_w_data[in_idx] * filters_w_data[filter_idx];
              }
            }
          }
          int out_idx = out_offset +
                        (filter_channel * out_height + out_y) * out_width +
                        out_x;
          res += biases_w_data[filter_channel];
          out_w_data[out_idx] = res;
        }
      }
    }
  }
}

void MathCpu::ConvDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &filters_w,
                        shared_ptr<Mat> &biases_w, shared_ptr<Mat> &out_w,
                        Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int num_input = params.num_input;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];
  int out_width = params.out_width;
  int out_height = params.out_height;

  float *in_w_data = &in_w->data_[0];
  float *in_dw_data = &in_w->dw_->data_[0];
  float *filters_w_data = &filters_w->data_[0];
  float *filters_dw_data = &filters_w->dw_->data_[0];
  float *biases_dw_data = &biases_w->dw_->data_[0];
  float *out_dw_data = &out_w->dw_->data_[0];

  for (int batch = 0; batch < batch_size; ++batch)
  {
    int in_offset = in_width * in_height * num_input * batch;
    int out_offset = out_width * out_height * num_filters * batch;

    for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
    {
      int filter_start_x = -padding_x;
      for (int out_x = 0; out_x < out_width;
           filter_start_x += stride_x, ++out_x)
      {
        int filter_start_y = -padding_y;
        for (int out_y = 0; out_y < out_height;
             filter_start_y += stride_y, ++out_y)
        {
          // grad
          int dw_idx = out_offset +
                       (filter_channel * out_height + out_y) * out_width +
                       out_x;
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
                int in_idx = in_offset +
                             (in_channel * in_height + in_y) * in_width + in_x;

                in_dw_data[in_idx] += filters_w_data[filter_idx] * dw;
                filters_dw_data[filter_idx] += in_w_data[in_idx] * dw;
              }
            }
          }
          biases_dw_data[filter_channel] += dw;
        }
      }
    }
  }
}

void MathCpu::MaxPool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                      Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int out_width = params.out_width;
  int out_height = params.out_height;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  float *in_w_data = &in_w->data_[0];
  float *out_w_data = &out_w->data_[0];

  for (int batch = 0; batch < batch_size; ++batch)
  {
    int in_offset = in_width * in_height * num_filters * batch;
    int out_offset = out_width * out_height * num_filters * batch;

    for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
    {
      int filter_start_x = -padding_x;
      for (int out_x = 0; out_x < out_width;
           filter_start_x += stride_x, ++out_x)
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

              int in_idx = in_offset +
                           (filter_channel * in_height + in_y) * in_width +
                           in_x;
              float curr = in_w_data[in_idx];
              if (curr > res)
              {
                res = curr;
              }
            }
          }

          int out_idx = out_offset +
                        (filter_channel * out_height + out_y) * out_width +
                        out_x;
          out_w_data[out_idx] = res;
        }
      }
    }
  }
}

void MathCpu::MaxPoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                           Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int out_width = params.out_width;
  int out_height = params.out_height;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  float *in_w_data = &in_w->data_[0];
  float *in_dw_data = &in_w->dw_->data_[0];
  float *out_dw_data = &out_w->dw_->data_[0];

  for (int batch = 0; batch < batch_size; ++batch)
  {
    int in_offset = in_width * in_height * num_filters * batch;
    int out_offset = out_width * out_height * num_filters * batch;

    for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
    {
      int filter_start_x = -padding_x;
      for (int out_x = 0; out_x < out_width;
           filter_start_x += stride_x, ++out_x)
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

              int in_idx = in_offset +
                           (filter_channel * in_height + in_y) * in_width +
                           in_x;
              float curr = in_w_data[in_idx];
              if (curr > res)
              {
                res = curr;
                res_idx = in_idx;
              }
            }
          }

          int out_idx = out_offset +
                        (filter_channel * out_height + out_y) * out_width +
                        out_x;
          in_dw_data[res_idx] += out_dw_data[out_idx];
        }
      }
    }
  }
}

// TODO, max pool for a while
void MathCpu::AvePool(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                      Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int out_width = params.out_width;
  int out_height = params.out_height;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  float *in_w_data = &in_w->data_[0];
  float *out_w_data = &out_w->data_[0];

  for (int batch = 0; batch < batch_size; ++batch)
  {
    int in_offset = in_width * in_height * num_filters * batch;
    int out_offset = out_width * out_height * num_filters * batch;

    for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
    {
      int filter_start_x = -padding_x;
      for (int out_x = 0; out_x < out_width;
           filter_start_x += stride_x, ++out_x)
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

              int in_idx = in_offset +
                           (filter_channel * in_height + in_y) * in_width +
                           in_x;
              float curr = in_w_data[in_idx];
              if (curr > res)
              {
                res = curr;
              }
            }
          }

          int out_idx = out_offset +
                        (filter_channel * out_height + out_y) * out_width +
                        out_x;
          out_w_data[out_idx] = res;
        }
      }
    }
  }
}

// TODO, max pool for a while
void MathCpu::AvePoolDeriv(shared_ptr<Mat> &in_w, shared_ptr<Mat> &out_w,
                           Params &params)
{
  int padding_x = params.padding_x;
  int padding_y = params.padding_y;
  int stride_x = params.stride_x;
  int stride_y = params.stride_y;
  int filter_width = params.filter_width;
  int filter_height = params.filter_height;
  int out_width = params.out_width;
  int out_height = params.out_height;
  int num_filters = params.num_output;
  int in_width = in_w->size_[0];
  int in_height = in_w->size_[1];
  int batch_size = in_w->size_[3];

  float *in_w_data = &in_w->data_[0];
  float *in_dw_data = &in_w->dw_->data_[0];
  float *out_dw_data = &out_w->dw_->data_[0];

  for (int batch = 0; batch < batch_size; ++batch)
  {
    int in_offset = in_width * in_height * num_filters * batch;
    int out_offset = out_width * out_height * num_filters * batch;

    for (int filter_channel = 0; filter_channel < num_filters; ++filter_channel)
    {
      int filter_start_x = -padding_x;
      for (int out_x = 0; out_x < out_width;
           filter_start_x += stride_x, ++out_x)
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

              int in_idx = in_offset +
                           (filter_channel * in_height + in_y) * in_width +
                           in_x;
              float curr = in_w_data[in_idx];
              if (curr > res)
              {
                res = curr;
                res_idx = in_idx;
              }
            }
          }

          int out_idx = out_offset +
                        (filter_channel * out_height + out_y) * out_width +
                        out_x;
          in_dw_data[res_idx] += out_dw_data[out_idx];
        }
      }
    }
  }
}

void MathCpu::SGD(shared_ptr<Mat> &mat, float learning_rate, int batch_size)
{
  for (size_t i = 0; i < mat->data_.size(); ++i)
  {
    if (mat->dw_->data_[i] != 0)
    {
      mat->data_[i] += -learning_rate * (mat->dw_->data_[i] / batch_size);
    }
  }
}

void MathCpu::Rmsprop(shared_ptr<Mat> &mat, shared_ptr<Mat> &mat_prev,
                      float learning_rate, int batch_size)
{
  float decay_rate = 0.999;
  float smooth_eps = 1e-8;
  float regc = 0.000001;  // L2 regularization strength
  float clipval = 5.0;    // clip gradients at this value

  for (size_t i = 0; i < mat->data_.size(); ++i)
  {
    // Rmsprop adaptive learning rate.
    float mdwi = mat->dw_->data_[i] / batch_size;
    mat_prev->data_[i] =
        decay_rate * mat_prev->data_[i] + (1.0 - decay_rate) * mdwi * mdwi;

    // Gradient clip.
    if (mdwi > clipval)
    {
      mdwi = clipval;
    }
    if (mdwi < -clipval)
    {
      mdwi = -clipval;
    }

    // Update (and regularize).
    mat->data_[i] +=
        -learning_rate * mdwi / sqrt(mat_prev->data_[i] + smooth_eps) -
        regc * mat->data_[i];

    mat->dw_->data_[i] = 0;
  }
}
