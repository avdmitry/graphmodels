#include "utils.h"
#include "layers.h"
#include "learn.h"
#include "datasets/mnist.h"

using std::string;
using std::vector;
using std::shared_ptr;

// 4.333 epoch| cost: 0.026| test acc: 0.979

class FcNet : public Model
{
 public:
  FcNet(int num_input, int num_output)
  {
    static const int num_hidden_units = 256;

    graph_ = shared_ptr<Graph>(new Graph);
    input_ = shared_ptr<MatWdw>(new MatWdw(num_input, 1));

    shared_ptr<MatWdw> a1mat, h1mat;
    graph_->Process(
        shared_ptr<Object>(new FCLayer(input_, &a1mat, num_hidden_units)));
    graph_->Process(shared_ptr<Object>(new ReluOp(a1mat, &h1mat)));

    shared_ptr<MatWdw> a2mat, h2mat;
    graph_->Process(
        shared_ptr<Object>(new FCLayer(h1mat, &a2mat, num_hidden_units)));
    graph_->Process(shared_ptr<Object>(new ReluOp(a2mat, &h2mat)));

    graph_->Process(
        shared_ptr<Object>(new FCLayer(h2mat, &output_, num_output)));

    graph_->GetParams(params_);
    for (size_t i = 0; i < params_.size(); ++i)
    {
      shared_ptr<MatWdw> &mat = params_[i];
      params_prev_.emplace_back(new MatWdw(mat->size_[0], mat->size_[1],
                                           mat->size_[2], mat->size_[3]));
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
  math->Init();

  datasets::Mnist mnist(data_path);
  vector<shared_ptr<datasets::MnistObj>> &train = mnist.train_;
  vector<shared_ptr<datasets::MnistObj>> &test = mnist.test_;
  printf("%lu\n", train.size());
  printf("%lu\n", test.size());

  shared_ptr<Model> net(new FcNet(784, 10));
  float learning_rate = 0.001;

  int epoch_num = 0;
  float cost = 0.0;
  clock_t begin_time = clock();
  for (int step = 0; step < 1000000; ++step)
  {
    int train_idx = step % train.size();
    // int train_idx = Random01()*(train.size()-1);
    for (int idx = 0; idx < 784; ++idx)
    {
      net->input_->w_->data_[idx] = train[train_idx]->image[idx];
    }
    int idx_target = train[train_idx]->label;

    net->Forward();
    // printf("output: %u %u %u %u\n", logprobs->size_[0], logprobs->size_[1],
    //         logprobs->size_[2], logprobs->size_[3]);

    cost += SoftmaxLoss(net, idx_target);

    net->Backward();

    LearnRmsprop(net, learning_rate);

    bool new_line = false;
    int output_each = 1000;
    int time_each = 1000;
    int validate_each = 10000;

    int epoch_num_curr = step / train.size();
    if (epoch_num != epoch_num_curr)
    {
      epoch_num = epoch_num_curr;
      learning_rate /= 2;
      printf("learning rate: %.6f\n", learning_rate);
    }
    if (step % output_each == 0 && step != 0)
    {
      printf("%.3f epoch| cost: %.3f", 1.0f * step / train.size(),
             cost / output_each);
      cost = 0.0;
      new_line = true;
    }
    if (step % validate_each == 0 && step != 0)
    {
      float acc = 0;
      for (int test_idx = 0; test_idx < test.size(); ++test_idx)
      {
        for (int idx = 0; idx < 784; ++idx)
        {
          net->input_->w_->data_[idx] = test[test_idx]->image[idx];
        }
        int idx_gt = test[test_idx]->label;

        net->Forward();

        int idx_pred = MaxIdx(net->output_->w_->data_);
        if (idx_gt == idx_pred)
        {
          acc += 1;
        }
      }
      acc /= test.size();

      printf("| test acc: %.3f", acc);
      new_line = true;
    }
    if (step % time_each == 0 && step != 0)
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
