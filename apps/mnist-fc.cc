#include "utils.h"
#include "layers.h"
#include "learn.h"
#include "datasets/mnist.h"

using std::string;
using std::vector;
using std::shared_ptr;

// bs_1: 4.333 epoch| cost: 0.026| test acc: 0.979
// bs_10: 2.500 epoch| cost: 0.003| test acc: 0.980
// bs_10: 4.333 epoch| cost: 0.001| test acc: 0.983

class FcNet : public Model
{
 public:
  FcNet(int num_input, int num_output, int batch_size)
  {
    static const int num_hidden_units = 256;

    graph_ = shared_ptr<Graph>(new Graph);
    input_ = shared_ptr<Mat>(new Mat(num_input, 1, 1, batch_size));

    shared_ptr<Mat> fc1, rel1;
    graph_->Process(
        shared_ptr<Object>(new FCLayer(input_, &fc1, num_hidden_units)));
    graph_->Process(shared_ptr<Object>(new ReluOp(fc1, &rel1)));

    shared_ptr<Mat> fc2, rel2;
    graph_->Process(
        shared_ptr<Object>(new FCLayer(rel1, &fc2, num_hidden_units)));
    graph_->Process(shared_ptr<Object>(new ReluOp(fc2, &rel2)));

    graph_->Process(
        shared_ptr<Object>(new FCLayer(rel2, &output_, num_output)));

    graph_->GetParams(params_);
    for (size_t i = 0; i < params_.size(); ++i)
    {
      shared_ptr<Mat> &mat = params_[i];
      params_prev_.emplace_back(new Mat(mat->size_));
    }
  }

  // Virtual functions stubs.
  void Create(int idx)
  {
  }

  void ClearPrevState()
  {
  }
};

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    printf("usage: mnist_data_path\n");
    return -1;
  }
  string data_path(argv[1]);

  // srand(time(NULL));
  srand(0);

  math = shared_ptr<Math>(new MathCpu);
  // math = shared_ptr<Math>(new MathCudnn);
  math->Init();

  datasets::Mnist mnist(data_path);
  vector<shared_ptr<datasets::MnistObj>> &train = mnist.train_;
  vector<shared_ptr<datasets::MnistObj>> &test = mnist.test_;
  printf("train: %lu\n", train.size());
  printf("test: %lu\n", test.size());

  int batch_size = 10;
  printf("epoch size in batches: %.1f\n", 1.0 * train.size() / batch_size);

  shared_ptr<Model> net(new FcNet(784, 10, batch_size));
  float learning_rate = 0.0005;

  int epoch_num = 0;
  int steps_num = 1000000;
  float cost = 0.0;
  int train_idx = 0;
  clock_t begin_time = clock();
  for (int step = 0; step < steps_num; ++step)
  {
    vector<int> idx_target;
    for (int batch = 0; batch < batch_size; ++batch)
    {
      for (int idx = 0; idx < 784; ++idx)
      {
        net->input_->data_[idx + batch * 784] = train[train_idx]->image[idx];
      }
      idx_target.emplace_back(train[train_idx]->label);

      train_idx++;
      if (train_idx == train.size())
      {
        train_idx = 0;
      }
    }
    net->Forward(true);
    // printf("output: %u %u %u %u\n", logprobs->size_[0], logprobs->size_[1],
    //         logprobs->size_[2], logprobs->size_[3]);

    cost += SoftmaxLoss(net, idx_target);

    net->Backward();

    LearnRmsprop(net, learning_rate, batch_size);

    bool new_line = false;
    int output_each = 1000;
    int time_each = 1000;
    int validate_each = 10000;
    int lr_drop_each = 2 * 60000;

    int num_examples = step * batch_size;

    int epoch_num_curr = num_examples / train.size();
    if (epoch_num != epoch_num_curr)
    {
      epoch_num = epoch_num_curr;
    }

    if (num_examples % lr_drop_each == 0 && step != 0)
    {
      learning_rate /= 2;
      printf("learning rate: %.6f\n", learning_rate);
    }

    if (num_examples % output_each == 0 && step != 0)
    {
      printf("%.3f epoch| cost: %.3f", 1.0f * num_examples / train.size(),
             cost / (output_each * batch_size));
      cost = 0.0;
      new_line = true;
    }
    if (num_examples % validate_each == 0 && step != 0)
    {
      net->SetBatchSize(1);
      float acc = 0;
      for (int test_idx = 0; test_idx < test.size(); ++test_idx)
      {
        for (int idx = 0; idx < 784; ++idx)
        {
          net->input_->data_[idx] = test[test_idx]->image[idx];
        }
        int idx_gt = test[test_idx]->label;

        net->Forward(false);

        int idx_pred = MaxIdx(net->output_);
        if (idx_gt == idx_pred)
        {
          acc += 1;
        }
      }
      acc /= test.size();
      net->SetBatchSize(batch_size);

      printf("| test acc: %.3f", acc);
      new_line = true;
    }
    if (num_examples % time_each == 0 && step != 0)
    {
      float time_curr = float(clock() - begin_time) / CLOCKS_PER_SEC;
      printf("| time: %.3f s", time_curr);
      begin_time = clock();
      new_line = true;
    }
    if (new_line)
    {
      printf("\n");
    }
  }

  return 0;
}
