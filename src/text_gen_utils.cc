#include "text_gen_utils.h"

using std::string;
using std::vector;
using std::shared_ptr;
using std::ifstream;
using std::set;

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

float CalcCost(shared_ptr<Model> &model, string &sent, shared_ptr<Data> &data)
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

    model->Create(idx_source);

    model->graph_->Forward(true, true);

    shared_ptr<Mat> labels(new Mat(1, 1, 1, 1, false));
    labels->data_[0] = idx_target;
    shared_ptr<Mat> out;
    cost += math->Softmax(model->output_, out, labels);
  }

  return cost / sent.length();
}

string PredictSentence(shared_ptr<Model> &model, shared_ptr<Data> &data,
                       bool sample_idx, float temperature)
{
  model->graph_ = shared_ptr<Graph>(new Graph);
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

    model->Create(idx);

    model->graph_->Forward(false, true);
    shared_ptr<Mat> logprobs = model->output_;

    // Sample predicted letter.
    if (temperature > 0.0 && temperature < 1.0 && sample_idx)
    {
      // Scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky.
      for (int i = 0; i < logprobs->data_.size(); i++)
      {
        logprobs->data_[i] /= temperature;
      }
    }

    shared_ptr<Mat> probs;
    shared_ptr<Mat> labels(new Mat(1, 1, 1, 1, false));  // Not used.
    math->Softmax(logprobs, probs, labels);
    if (sample_idx)
    {
      idx = SampleIdx(probs->data_);
    }
    else
    {
      idx = MaxIdx(probs);
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
