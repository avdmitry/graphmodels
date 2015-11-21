#ifndef LEARN_H
#define LEARN_H

#include "utils.h"

void LearnSGD(std::shared_ptr<Model> &model, float learning_rate,
              int batch_size = 1);

void LearnRmsprop(std::shared_ptr<Model> &model, float learning_rate,
                  int batch_size = 1);

#endif
