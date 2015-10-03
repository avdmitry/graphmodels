#include "learn.h"

using std::shared_ptr;

void LearnSGD(shared_ptr<Model> &model)
{
  float learning_rate = 0.01;
  for (size_t j = 0; j < model->params_.size(); ++j)
  {
    shared_ptr<MatWdw> &mat = model->params_[j];
    // shared_ptr<MatWdw> &mat_prev = model->params_prev_[j];
    for (size_t i = 0; i < mat->w_->data_.size(); ++i)
    {
      if (mat->dw_->data_[i] != 0)
      {
        mat->w_->data_[i] += -learning_rate * mat->dw_->data_[i];
      }
    }
  }

  std::fill(model->output_->dw_->data_.begin(),
            model->output_->dw_->data_.end(), 0);
  model->graph_->ClearDw();
}

void LearnRmsprop(shared_ptr<Model> &model)
{
  float decay_rate = 0.999;
  float smooth_eps = 1e-8;
  float regc = 0.000001;        // L2 regularization strength
  float learning_rate = 0.001;  // learning rate, 0.01 for lstm, 0.001 for rnn
  float clipval = 5.0;          // clip gradients at this value
  for (size_t j = 0; j < model->params_.size(); ++j)
  {
    shared_ptr<MatWdw> &mat = model->params_[j];
    shared_ptr<MatWdw> &mat_prev = model->params_prev_[j];
    for (size_t i = 0; i < mat->w_->data_.size(); ++i)
    {
      // Rmsprop adaptive learning rate.
      float mdwi = mat->dw_->data_[i];
      mat_prev->w_->data_[i] = decay_rate * mat_prev->w_->data_[i] +
                               (1.0 - decay_rate) * mdwi * mdwi;

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
      mat->w_->data_[i] +=
          -learning_rate * mdwi / sqrt(mat_prev->w_->data_[i] + smooth_eps) -
          regc * mat->w_->data_[i];
      mat->dw_->data_[i] = 0;
    }
  }
}
