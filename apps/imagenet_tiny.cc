#include "utils.h"
#include "layers.h"
#include "learn.h"

#include <algorithm>  // partial_sort
#include <random>     // std::default_random_engine
#include <chrono>     // std::chrono::system_clock

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using std::string;
using std::vector;
using std::shared_ptr;

using cv::Size;

inline void ResizeOCV(cv::Mat &img, unsigned int width, unsigned int height)
{
  resize(img, img, Size(width, height), 0, 0, cv::INTER_LINEAR);
}

inline void RotateOCV(cv::Mat &img, float angle)
{
  cv::Mat rot =
      getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), angle, 1.0);
  cv::Mat out;
  warpAffine(img, out, rot, Size(img.cols, img.rows));
  img = out;
}

inline void CropOCV(cv::Mat &img, cv::Mat &out, int left, int top, int right,
                    int bottom)
{
  cv::Mat ref(img, cv::Rect(left, top, right - left, bottom - top));
  ref.copyTo(out);
}

inline void CropOCV(cv::Mat &img, int left, int top, int right, int bottom)
{
  cv::Mat out;
  CropOCV(img, out, left, top, right, bottom);
  img = out;
}

inline void MirrorOCV(cv::Mat &img)
{
  cv::Mat out;
  flip(img, out, 1);  // 0 - x, 1 - y, -1 - both
  img = out;
}

inline unsigned int SpectrumOCV(cv::Mat &img)
{
  return 1 + (img.type() >> CV_CN_SHIFT);
}

void Fill(cv::Mat &image, float *data)
{
  int num_image_colors = SpectrumOCV(image);
  int num_pixels = image.cols * image.rows;
  if (num_image_colors == 3)
  {
    // Convert from opencv Mat to format: "rr..gg..bb".
    for (int j = 0, posr = 0; j < image.rows; ++j, posr += image.cols)
    {
      unsigned int offset0 = posr;
      unsigned int offset1 = num_pixels + posr;
      unsigned int offset2 = 2 * num_pixels + posr;
      unsigned char *imgr = image.ptr<unsigned char>(j);
      for (int k = 0, posc = 0; k < image.cols; ++k, posc += 3)
      {
        data[offset0 + k] = 1.0 * imgr[posc + 2] / 255 - 0.5;  // -0.5-0.5
        data[offset1 + k] = 1.0 * imgr[posc + 1] / 255 - 0.5;  // -0.5-0.5
        data[offset2 + k] = 1.0 * imgr[posc] / 255 - 0.5;      // -0.5-0.5
      }
    }
  }
}

class CnnNet : public Model
{
 public:
  CnnNet(int num_input_x, int num_input_y, int num_output, int batch_size,
         int input_channels = 3)
  {
    graph_ = shared_ptr<Graph>(new Graph);
    input_ = shared_ptr<Mat>(
        new Mat(num_input_x, num_input_y, input_channels, batch_size));
    math->MemoryAlloc(input_);
    math->MemoryAlloc(input_->dw_);

    {
      static const int filter0_x = 7;
      static const int filter0_y = 7;
      static const int num_filters0 = 64;
      shared_ptr<Mat> mp0, mp1, mp2, mp3, pool;
      shared_ptr<Mat> conv0, rel0, conv1, rel1, conv2, rel2, conv3, rel3, conv4,
          rel4;
      shared_ptr<Mat> conv12, rel12, conv22, rel22, conv32, rel32, conv42,
          rel42;
      shared_ptr<Mat> conv13, rel13, conv23, rel23, conv33, rel33, conv43,
          rel43;
      shared_ptr<Mat> conv14, rel14, conv24, rel24, conv34, rel34, conv44,
          rel44;
      shared_ptr<Mat> bn0, bn1, bn2, bn3, bn4;
      shared_ptr<Mat> bn12, bn22, bn32, bn42;
      shared_ptr<Mat> bn13, bn23, bn33, bn43;
      shared_ptr<Mat> bn14, bn24, bn34, bn44;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv0", input_, &conv0, num_filters0, filter0_x,
                        filter0_y, 3, 3, 2, 2)));
      graph_->Process(
          shared_ptr<Operation>(new BatchNormOp("bn0", conv0, &bn0)));
      graph_->Process(shared_ptr<Operation>(new ReluOp(bn0, &rel0)));
      graph_->Process(shared_ptr<Operation>(
          new PoolLayer(rel0, &mp0, 3, 3, 1, 1, 2, 2, MAX)));

      static const int filter1_x = 3;
      static const int filter1_y = 3;
      static const int num_filters1 = 128;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv1", mp0, &conv1, num_filters1, filter1_x,
                        filter1_y, 1, 1, 1, 1)));
      graph_->Process(
          shared_ptr<Operation>(new BatchNormOp("bn1", conv1, &bn1)));
      graph_->Process(shared_ptr<Operation>(new ReluOp(bn1, &rel1)));
      graph_->Process(shared_ptr<Operation>(
          new PoolLayer(rel1, &mp1, 3, 3, 1, 1, 2, 2, MAX)));

      static const int filter2_x = 3;
      static const int filter2_y = 3;
      static const int num_filters2 = 256;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv2", mp1, &conv2, num_filters2, filter2_x,
                        filter2_y, 1, 1, 1, 1)));
      graph_->Process(
          shared_ptr<Operation>(new BatchNormOp("bn2", conv2, &bn2)));
      graph_->Process(shared_ptr<Operation>(new ReluOp(bn2, &rel2)));
      graph_->Process(shared_ptr<Operation>(
          new PoolLayer(rel2, &mp2, 3, 3, 1, 1, 2, 2, MAX)));

      static const int filter3_x = 3;
      static const int filter3_y = 3;
      static const int num_filters3 = 512;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv3", mp2, &conv3, num_filters3, filter3_x,
                        filter3_y, 1, 1, 1, 1)));
      graph_->Process(
          shared_ptr<Operation>(new BatchNormOp("bn3", conv3, &bn3)));
      graph_->Process(shared_ptr<Operation>(new ReluOp(bn3, &rel3)));

      graph_->Process(shared_ptr<Operation>(
          new PoolLayer(rel3, &mp3, 3, 3, 1, 1, 2, 2, MAX)));

      static const int filter4_x = 3;
      static const int filter4_y = 3;
      static const int num_filters4 = 512;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv4", mp3, &conv4, num_filters4, filter4_x,
                        filter4_y, 1, 1, 1, 1)));
      graph_->Process(
          shared_ptr<Operation>(new BatchNormOp("bn4", conv4, &bn4)));
      graph_->Process(shared_ptr<Operation>(new ReluOp(bn4, &rel4)));

      static const int filter42_x = 3;
      static const int filter42_y = 3;
      static const int num_filters42 = 512;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv42", rel4, &conv42, num_filters42, filter42_x,
                        filter42_y, 1, 1, 1, 1)));
      graph_->Process(
          shared_ptr<Operation>(new BatchNormOp("bn4", conv42, &bn42)));
      graph_->Process(shared_ptr<Operation>(new ReluOp(bn42, &rel42)));

      static const int filter5_x = 1;
      static const int filter5_y = 1;
      static const int num_filters5 = 1000;
      graph_->Process(shared_ptr<Operation>(
          new ConvLayer("conv5", rel42, &pool, num_filters5, filter5_x,
                        filter5_y, 0, 0, 1, 1)));
      graph_->Process(shared_ptr<Operation>(
          new PoolLayer(pool, &output_, 7, 7, 0, 0, 7, 7, AVE)));
    }

    graph_->GetParams(params_);
    for (size_t i = 0; i < params_.size(); ++i)
    {
      shared_ptr<Mat> &mat = params_[i];
      params_prev_.emplace_back(new Mat(mat->size_, false));
      math->CopyToDevice(params_prev_.back());
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

struct ImagenetObj
{
  string path;
  int label;
};

void Validate(shared_ptr<Model> &net, vector<shared_ptr<ImagenetObj>> &test)
{
  int batch_size = net->output_->size_[3];
  float acc = 0, acc5 = 0;
  int test_idx = 0;
  shared_ptr<Mat> labels(new Mat(1, 1, 1, batch_size, false));
  while (test_idx < test.size())
  {
    int curr_size = batch_size;
    for (int batch = 0; batch < batch_size; ++batch)
    {
      shared_ptr<ImagenetObj> &curr = test[test_idx];
      cv::Mat img = cv::imread(curr->path);
      int dim = img.cols;
      if (img.rows < dim)
      {
        dim = img.rows;
      }
      ResizeOCV(img, 256.0 * img.cols / dim, 256.0 * img.rows / dim);
      int left = (img.cols - 224) / 2;
      int top = (img.rows - 224) / 2;
      CropOCV(img, left, top, left + 224, top + 224);
      Fill(img, &net->input_->data_[batch * 224 * 224 * 3]);

      labels->data_[batch] = curr->label;

      test_idx++;
      if (test_idx % 1000 == 0)
      {
        printf("validated: %u\n", test_idx);
      }
      if (test_idx == test.size())
      {
        curr_size = batch + 1;
        break;
      }
    }

    math->CopyToDevice(net->input_);
    net->Forward(false);
    math->CopyToHost(net->output_);

    struct ValueIndex
    {
      float value;
      int index;
    };
    vector<ValueIndex> curr(1000);
    for (int batch = 0; batch < curr_size; ++batch)
    {
      float *data = &net->output_->data_[batch * 1000];
      int pred_idx = 0;
      for (int prob_idx = 0; prob_idx < 1000; ++prob_idx)
      {
        curr[prob_idx].value = data[prob_idx];
        curr[prob_idx].index = prob_idx;
        if (data[pred_idx] < data[prob_idx])
        {
          pred_idx = prob_idx;
        }
      }
      auto GreaterComp = [](ValueIndex i, ValueIndex j)
      {
        return (i.value > j.value);
      };
      std::partial_sort(curr.begin(), curr.begin() + 5, curr.end(),
                        GreaterComp);
      if (labels->data_[batch] == curr[0].index)
      {
        acc += 1;
      }
      for (int idx = 0; idx < 5; ++idx)
      {
        if (labels->data_[batch] == curr[idx].index)
        {
          acc5 += 1;
          break;
        }
      }
    }
  }

  printf("test acc, top1: %.3f, top-5: %.3f\n", acc / test.size(),
         acc5 / test.size());
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("usage: imagenet_data_path [model]\n");
    return -1;
  }
  string data_path(argv[1]);

  string model_name;
  if (argc > 2)
  {
    model_name = argv[2];
  }

  // srand(time(NULL));
  srand(0);

  // math = shared_ptr<Math>(new MathCpu);
  math = shared_ptr<Math>(new MathCudnn(0));
  math->Init();

  vector<shared_ptr<ImagenetObj>> train, test;
  string line;
  std::fstream train_file(data_path + "train.txt");
  while (getline(train_file, line))
  {
    shared_ptr<ImagenetObj> curr(new ImagenetObj);
    shared_ptr<vector<string>> elems = Split(line, ' ');
    curr->path = (*elems)[0];
    curr->label = std::stoi((*elems)[1]);
    train.emplace_back(curr);
  }
  unsigned int seed = 0;
  shuffle(train.begin(), train.end(), std::default_random_engine(seed));

  std::fstream val_file(data_path + "val.txt");
  while (getline(val_file, line))
  {
    shared_ptr<ImagenetObj> curr(new ImagenetObj);
    shared_ptr<vector<string>> elems = Split(line, ' ');
    curr->path = (*elems)[0];
    curr->label = std::stoi((*elems)[1]);
    test.emplace_back(curr);
  }
  printf("train: %lu\n", train.size());
  printf("test: %lu\n", test.size());

  // Hyperparameters.
  float learning_rate = 0.1;  // sgd
  float decay_rate = 0.0005;
  int batch_size = 64;
  int output_each = batch_size * 10;
  int time_each = batch_size * 10;
  int validate_each = batch_size * 5000;
  int save_each = batch_size * 1000;
  // int lr_drop_each = batch_size * 150000;
  // int steps_num = 450000;
  int lr_drop_each = batch_size * 40000;
  int steps_num = 160000;
  int start_step = 0;
  int output_each_curr = 0;
  int time_each_curr = 0;
  int save_each_curr = 0;
  int lr_drop_each_curr = 0;

  printf("epoch size in batches: %.1f\n", 1.0 * train.size() / batch_size);

  shared_ptr<Model> net(new CnnNet(224, 224, 1000, batch_size));

  if (model_name.length() > 0)
  {
    printf("loading %s\n", model_name.c_str());
    FILE *file = fopen(model_name.c_str(), "rb");
    net->graph_->Load(file);
    int res = fread((void *)&start_step, sizeof(int), 1, file);
    printf("start_step: %u\n", start_step);
    res = fread((void *)&learning_rate, sizeof(float), 1, file);
    printf("learning_rate: %f\n", learning_rate);
    res = fread((void *)&output_each_curr, sizeof(int), 1, file);
    printf("output_each_curr: %u\n", output_each_curr);
    res = fread((void *)&time_each_curr, sizeof(int), 1, file);
    printf("time_each_curr: %u\n", time_each_curr);
    res = fread((void *)&save_each_curr, sizeof(int), 1, file);
    printf("save_each_curr: %u\n", save_each_curr);
    res = fread((void *)&lr_drop_each_curr, sizeof(int), 1, file);
    printf("lr_drop_each_curr: %u\n", lr_drop_each_curr);

    fclose(file);

    Validate(net, test);
    exit(0);
  }

  int num_examples = 0;
  int epoch_num = 0;
  float cost = 0.0;
  int train_idx = 0;
  clock_t begin_time = clock();
  for (int step = start_step + 1; step <= steps_num; ++step)
  {
    shared_ptr<Mat> labels(new Mat(1, 1, 1, batch_size, false));
    for (int batch = 0; batch < batch_size; ++batch)
    {
      shared_ptr<ImagenetObj> &curr = train[train_idx];
      cv::Mat img = cv::imread(curr->path);
      int dim = img.cols;
      if (img.rows < dim)
      {
        dim = img.rows;
      }
      ResizeOCV(img, 256.0 * img.cols / dim, 256.0 * img.rows / dim);
      int delta_left = (img.cols - 224) / 2;
      int delta_top = (img.rows - 224) / 2;
      int left = Randi(0, delta_left);
      int top = Randi(0, delta_top);
      CropOCV(img, left, top, left + 224, top + 224);
      Fill(img, &net->input_->data_[batch * 224 * 224 * 3]);

      labels->data_[batch] = curr->label;

      train_idx++;
      if (train_idx == train.size())
      {
        train_idx = 0;
        seed += 1;
        shuffle(train.begin(), train.end(), std::default_random_engine(seed));
      }
    }
    math->CopyToDevice(net->input_);
    math->CopyToDevice(labels);

    net->Forward(true);

    shared_ptr<Mat> out;
    cost += math->Softmax(net->output_, out, labels);

    net->Backward();

    LearnSGD(net, learning_rate, batch_size, decay_rate);

    math->Sync();

    bool new_line = false;
    num_examples = step * batch_size;

    int epoch_num_curr = num_examples / train.size();
    if (epoch_num != epoch_num_curr)
    {
      epoch_num = epoch_num_curr;
    }

    lr_drop_each_curr += batch_size;
    if (lr_drop_each_curr >= lr_drop_each)
    {
      lr_drop_each_curr = 0;
      learning_rate *= 0.1;
      printf("learning rate: %.6f\n", learning_rate);
    }

    save_each_curr += batch_size;
    if (save_each_curr >= save_each)
    {
      save_each_curr = 0;
      string file_name("imagenet_" + std::to_string(step) + ".model");
      printf("saving %s\n", file_name.c_str());
      FILE *file = fopen(file_name.c_str(), "wb");
      net->graph_->Save(file);
      fwrite((void *)&step, sizeof(int), 1, file);
      fwrite((void *)&learning_rate, sizeof(float), 1, file);
      fwrite((void *)&output_each_curr, sizeof(int), 1, file);
      fwrite((void *)&time_each_curr, sizeof(int), 1, file);
      fwrite((void *)&save_each_curr, sizeof(int), 1, file);
      fwrite((void *)&lr_drop_each_curr, sizeof(int), 1, file);
      fclose(file);
    }

    output_each_curr += batch_size;
    if (output_each_curr >= output_each)
    {
      output_each_curr = 0;
      printf("%.3f epoch| cost: %.6f", 1.0f * num_examples / train.size(),
             cost / (output_each /* * batch_size*/));
      cost = 0.0;
      new_line = true;
    }
    if (num_examples % validate_each == 0 && step != 0)
    {
      Validate(net, test);
      new_line = true;
    }
    time_each_curr += batch_size;
    if (time_each_curr >= time_each)
    {
      time_each_curr = 0;
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
