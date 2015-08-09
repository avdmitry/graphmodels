#include "text_gen_utils.h"

using namespace std;

void LoadData(const string &file_name, shared_ptr<Data> &data)
{
  string line;
  ifstream infile(file_name);
  while (getline(infile, line))
  {
    Trim(&line);
    if (line.length() == 0)
    {
      continue;
    }
    for (size_t i = 0; i < line.length(); ++i)
    {
      char ch = line[i];
      data->vocab_.insert(ch);
    }
    data->sentences_.emplace_back(line);
  }

  printf("num sentences: %lu, vocab: %lu\n", data->sentences_.size(),
         data->vocab_.size());
  int i = 1;
  for (set<char>::iterator it = data->vocab_.begin(); it != data->vocab_.end();
       ++it)
  {
    data->letter_to_index_[*it] = i;
    data->index_to_letter_[i] = *it;
    i += 1;

    // printf("%c %u\n", *it, data->letterToIndex[*it]);
  }
}

void Learn(shared_ptr<Model> &model)
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

float CalcCost(shared_ptr<Graph> &graph, shared_ptr<Model> &model, string &sent,
               shared_ptr<Data> &data)
{
  model->ClearPrevState();

  float cost = 0.0;
  int len = sent.length();
  for (int i = -1; i < len; ++i)
  {
    int idx_source = 0;
    int idx_target = 0;
    if (i != -1)
    {
      idx_source = data->letter_to_index_[sent[i]];
    }
    if (i != len - 1)
    {
      idx_target = data->letter_to_index_[sent[i + 1]];
    }

    model->Create(graph, idx_source);

    graph->Forward();
    shared_ptr<MatWdw> logprobs = model->output_;

    shared_ptr<Mat> probs = math->Softmax(logprobs->w_);
    cost += -log(probs->data_[idx_target]);

    // Write gradients into log probabilities.
    logprobs->dw_->data_ = probs->data_;
    logprobs->dw_->data_[idx_target] -= 1;
  }

  return cost / sent.length();
}

string PredictSentence(shared_ptr<Model> &model, shared_ptr<Data> &data,
                       bool sample_idx, float temperature)
{
  shared_ptr<Graph> graph(new Graph);
  string sent;

  model->ClearPrevState();

  while (true)
  {
    int idx;
    if (sent.length() == 0)
    {
      idx = 0;
    }
    else
    {
      idx = data->letter_to_index_[sent[sent.length() - 1]];
    }

    model->Create(graph, idx);

    graph->Forward();
    shared_ptr<MatWdw> logprobs = model->output_;

    // Sample predicted letter.
    if (temperature > 0.0 && temperature < 1.0 && sample_idx)
    {
      // Scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky.
      for (int i = 0; i < logprobs->w_->data_.size(); i++)
      {
        logprobs->w_->data_[i] /= temperature;
      }
    }

    shared_ptr<Mat> probs = math->Softmax(logprobs->w_);
    if (sample_idx)
    {
      idx = SampleIdx(probs->data_);
    }
    else
    {
      idx = MaxIdx(probs->data_);
    }

    // End token predicted, break out.
    if (idx == 0)
    {
      break;
    }
    int max_chars_gen = 70;
    if (sent.length() > max_chars_gen)
    {
      break;
    }

    sent += data->index_to_letter_[idx];
  }

  return sent;
}
