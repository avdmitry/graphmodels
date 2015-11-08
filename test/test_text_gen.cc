#include "utils.h"
#include "rnn.h"
#include "lstm.h"
#include "text_gen_utils.h"
#include "learn.h"

using std::string;
using std::vector;
using std::shared_ptr;
using std::ifstream;

string rnn_out = "";

int main(int argc, char *argv[])
{
  srand(6);

  // math = shared_ptr<Math>(new MathCuda(0));
  // math = shared_ptr<Math>(new MathBlas);
  math = shared_ptr<Math>(new MathCpu);
  math->Init();

  if (argc != 4)
  {
    printf("usage: file_with_sentences lstm/rnn file_expected_output\n");
    return -1;
  }
  string data_file_name(argv[1]);
  string model_type(argv[2]);
  string expected_output_file_name(argv[3]);

  string expected_output;
  string line;
  ifstream infile(expected_output_file_name);
  while (getline(infile, line))
  {
    expected_output += line;
    expected_output += "\n";
  }

  shared_ptr<Data> data(new Data);
  LoadData(data_file_name, data);

  // Init model.
  vector<int> hs = {20, 20};
  int inout_size = data->vocab_.size() + 1;
  shared_ptr<Model> model;
  if (model_type == "rnn")
  {
    model = shared_ptr<Model>(new Rnn(kEmbedSize, hs, inout_size));
  }
  else if (model_type == "lstm")
  {
    model = shared_ptr<Model>(new Lstm(kEmbedSize, hs, inout_size));
  }
  else
  {
    printf("unknown model type\n");
    return -1;
  }

  string output;
  float cost_epoch = 0;
  int num_epoch = 0;
  clock_t begin_time = clock();
  for (int step = 0; step < 2 * data->sentences_.size() + 1; ++step)
  {
    string sent = data->sentences_[Randi(0, data->sentences_.size() - 1)];

    model->graph_ = shared_ptr<Graph>(new Graph);
    cost_epoch += CalcCost(model, sent, data);

    model->graph_->Backward();

    LearnRmsprop(model, 0.001);

    if (step % data->sentences_.size() == 0 && step != 0)
    {
      float time_epoch = float(clock() - begin_time) / CLOCKS_PER_SEC;
      cost_epoch /= data->sentences_.size();
      num_epoch += 1;
      printf("%u epoch, cost: %.3f, time: %.3f s\n", num_epoch, cost_epoch,
             time_epoch);
      char tmp[10];
      sprintf(tmp, "%.6f\n", cost_epoch);
      output += tmp;
      cost_epoch = 0;

      string predict;
      vector<bool> sample_type = {true, true, true, true, true, false};
      for (int i = 0; i < sample_type.size(); ++i)
      {
        predict = PredictSentence(model, data, sample_type[i]);
        output += predict;
        output += "\n";
      }

      begin_time = clock();
    }
  }

  if (output != expected_output)
  {
    printf("test failed\n");
    printf("expected:\n%s\n", expected_output.c_str());
    printf("output:\n%s\n", output.c_str());
  }
  else
  {
    printf("test passed\n");
  }

  return 0;
}
