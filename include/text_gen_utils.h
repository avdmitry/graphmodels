#ifndef TEXT_GEN_UTILS_H
#define TEXT_GEN_UTILS_H

#include "utils.h"
#include "rnn.h"
#include "lstm.h"

static const int kEmbedSize = 5;

void LoadData(const std::string &file_name, std::shared_ptr<Data> &data);

float CalcCost(std::shared_ptr<Model> &model, std::string &sent,
               std::shared_ptr<Data> &data);

std::string PredictSentence(std::shared_ptr<Model> &model,
                            std::shared_ptr<Data> &data,
                            bool sample_idx = false, float temperature = 1.0);

#endif
