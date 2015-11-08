#ifndef MNIST_H
#define MNIST_H

#include <memory>
#include <string>
#include <vector>

namespace datasets
{
class MnistObj
{
 public:
  int label;
  std::vector<float> image;
};

class Mnist
{
 public:
  Mnist(const std::string &path)
  {
    std::string train_images_file(path + "/train-images-idx3-ubyte");
    std::string train_labels_file(path + "/train-labels-idx1-ubyte");
    LoadDatasetPart(train_images_file, train_labels_file, 60000, train_);

    std::string test_images_file(path + "/t10k-images-idx3-ubyte");
    std::string test_labels_file(path + "/t10k-labels-idx1-ubyte");
    LoadDatasetPart(test_images_file, test_labels_file, 10000, test_);
  }

  std::vector<std::shared_ptr<MnistObj>> train_;
  std::vector<std::shared_ptr<MnistObj>> test_;

 private:
  void LoadDatasetPart(const std::string &images_file,
                       const std::string &labels_file, unsigned int num,
                       std::vector<std::shared_ptr<MnistObj>> &dataset)
  {
    static const int mnist_size = 28;
    static const int image_size = mnist_size * mnist_size;

    FILE *f = fopen(images_file.c_str(), "rb");
    fseek(f, 16, SEEK_CUR);
    std::vector<unsigned char> images(num * image_size);
    size_t res = fread(&images[0], 1, num * image_size, f);
    fclose(f);
    if (num * image_size != res)
    {
      return;
    }

    f = fopen(labels_file.c_str(), "rb");
    fseek(f, 8, SEEK_CUR);
    std::vector<char> labels(num);
    res = fread(&labels[0], 1, num, f);
    fclose(f);
    if (num != res)
    {
      return;
    }

    for (int i = 0; i < num; ++i)
    {
      std::shared_ptr<MnistObj> curr(new MnistObj);
      curr->label = labels[i];

      curr->image.resize(image_size);
      int imageIdx = i * image_size;
      for (int j = 0; j < mnist_size; ++j)
      {
        for (int k = 0; k < mnist_size; ++k)
        {
          int idx = j * mnist_size + k;
          curr->image[idx] = 1.0 * images[imageIdx + idx] / 255.0;  // - 0.5f;
        }
      }

      dataset.emplace_back(curr);
    }
  }
};
}

#endif
