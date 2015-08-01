#include "lstm.h"

using namespace std;

lstm::lstm(int input_size, vector<int> hidden_sizes, int output_size)
    : hidden_sizes_(hidden_sizes)
{
  int hidden_size;
  for (size_t d = 0; d < hidden_sizes_.size(); d++)
  {
    int prev_size;
    if (d == 0)
    {
      prev_size = input_size;
    }
    else
    {
      prev_size = hidden_sizes_[d - 1];
    }
    hidden_size = hidden_sizes_[d];

    // Gates params.
    wix_.emplace_back(RandMat(hidden_size, prev_size, -0.08, 0.08));
    wih_.emplace_back(RandMat(hidden_size, hidden_size, -0.08, 0.08));
    bi_.emplace_back(new Mat(hidden_size, 1));
    wfx_.emplace_back(RandMat(hidden_size, prev_size, -0.08, 0.08));
    wfh_.emplace_back(RandMat(hidden_size, hidden_size, -0.08, 0.08));
    bf_.emplace_back(new Mat(hidden_size, 1));
    wox_.emplace_back(RandMat(hidden_size, prev_size, -0.08, 0.08));
    woh_.emplace_back(RandMat(hidden_size, hidden_size, -0.08, 0.08));
    bo_.emplace_back(new Mat(hidden_size, 1));

    // Cell write params.
    wcx_.emplace_back(RandMat(hidden_size, prev_size, -0.08, 0.08));
    wch_.emplace_back(RandMat(hidden_size, hidden_size, -0.08, 0.08));
    bc_.emplace_back(new Mat(hidden_size, 1));
  }

  // Decoder params.
  whd_ = RandMat(output_size, hidden_size, -0.08, 0.08);
  bd_ = shared_ptr<Mat>(new Mat(output_size, 1));

  wil_ = RandMat(output_size, input_size, -0.08, 0.08);

  GetParameters(params_);
  for (size_t i = 0; i < params_.size(); ++i)
  {
    shared_ptr<Mat> &mat = params_[i];
    params_prev_.emplace_back(new Mat(mat->n_, mat->d_));
  }
}

shared_ptr<Mat> lstm::Forward(shared_ptr<Graph> &graph, int idx)
{
  if (prev_hiddens_.size() == 0)
  {
    for (size_t d = 0; d < hidden_sizes_.size(); d++)
    {
      prev_hiddens_.emplace_back(new Mat(hidden_sizes_[d], 1));
    }
  }
  if (prev_cells_.size() == 0)
  {
    for (size_t d = 0; d < hidden_sizes_.size(); d++)
    {
      prev_cells_.emplace_back(new Mat(hidden_sizes_[d], 1));
    }
  }

  shared_ptr<Mat> x;
  graph->Process(shared_ptr<Object>(new ExtractRowOp(wil_, idx, &x)));

  vector<shared_ptr<Mat>> hidden;
  vector<shared_ptr<Mat>> cell;
  for (size_t d = 0; d < hidden_sizes_.size(); d++)
  {
    shared_ptr<Mat> input_vector;
    if (d == 0)
    {
      input_vector = x;
    }
    else
    {
      input_vector = hidden[d - 1];
    }

    shared_ptr<Mat> hidden_prev = prev_hiddens_[d];
    shared_ptr<Mat> cell_prev = prev_cells_[d];

    // Input gate.
    shared_ptr<Mat> hi0, hi1;
    graph->Process(shared_ptr<Object>(new MulOp(wix_[d], input_vector, &hi0)));
    graph->Process(shared_ptr<Object>(new MulOp(wih_[d], hidden_prev, &hi1)));
    shared_ptr<Mat> hi;
    graph->Process(shared_ptr<Object>(new AddOp(hi0, hi1, &hi)));
    shared_ptr<Mat> biasi;
    graph->Process(shared_ptr<Object>(new AddOp(hi, bi_[d], &biasi)));
    shared_ptr<Mat> input_gate;
    graph->Process(shared_ptr<Object>(new SigmOp(biasi, &input_gate)));

    // Forget gate.
    shared_ptr<Mat> hf0, hf1;
    graph->Process(shared_ptr<Object>(new MulOp(wfx_[d], input_vector, &hf0)));
    graph->Process(shared_ptr<Object>(new MulOp(wfh_[d], hidden_prev, &hf1)));
    shared_ptr<Mat> hf;
    graph->Process(shared_ptr<Object>(new AddOp(hf0, hf1, &hf)));
    shared_ptr<Mat> biasf;
    graph->Process(shared_ptr<Object>(new AddOp(hf, bf_[d], &biasf)));
    shared_ptr<Mat> forget_gate;
    graph->Process(shared_ptr<Object>(new SigmOp(biasf, &forget_gate)));

    // Output gate.
    shared_ptr<Mat> ho0, ho1;
    graph->Process(shared_ptr<Object>(new MulOp(wox_[d], input_vector, &ho0)));
    graph->Process(shared_ptr<Object>(new MulOp(woh_[d], hidden_prev, &ho1)));
    shared_ptr<Mat> ho;
    graph->Process(shared_ptr<Object>(new AddOp(ho0, ho1, &ho)));
    shared_ptr<Mat> biaso;
    graph->Process(shared_ptr<Object>(new AddOp(ho, bo_[d], &biaso)));
    shared_ptr<Mat> output_gate;
    graph->Process(shared_ptr<Object>(new SigmOp(biaso, &output_gate)));

    // Write operation on cells.
    shared_ptr<Mat> hw0, hw1;
    graph->Process(shared_ptr<Object>(new MulOp(wcx_[d], input_vector, &hw0)));
    graph->Process(shared_ptr<Object>(new MulOp(wch_[d], hidden_prev, &hw1)));
    shared_ptr<Mat> hw;
    graph->Process(shared_ptr<Object>(new AddOp(hw0, hw1, &hw)));
    shared_ptr<Mat> biasw;
    graph->Process(shared_ptr<Object>(new AddOp(hw, bc_[d], &biasw)));
    shared_ptr<Mat> cell_write;
    graph->Process(shared_ptr<Object>(new TanhOp(biasw, &cell_write)));

    // Compute new cell activation.
    // What do we keep from cell.
    shared_ptr<Mat> retain_cell;
    graph->Process(
        shared_ptr<Object>(new EltMulOp(forget_gate, cell_prev, &retain_cell)));
    // What do we write to cell.
    shared_ptr<Mat> write_cell;
    graph->Process(
        shared_ptr<Object>(new EltMulOp(input_gate, cell_write, &write_cell)));
    // New cell contents.
    shared_ptr<Mat> cell_curr;
    graph->Process(
        shared_ptr<Object>(new AddOp(retain_cell, write_cell, &cell_curr)));

    // Compute hidden state as gated, saturated cell activations.
    shared_ptr<Mat> tanhc;
    graph->Process(shared_ptr<Object>(new TanhOp(cell_curr, &tanhc)));
    shared_ptr<Mat> hidden_curr;
    graph->Process(
        shared_ptr<Object>(new EltMulOp(output_gate, tanhc, &hidden_curr)));

    cell.emplace_back(cell_curr);
    hidden.emplace_back(hidden_curr);
  }

  // Decoder.
  shared_ptr<Mat> hd;
  graph->Process(
      shared_ptr<Object>(new MulOp(whd_, hidden[hidden.size() - 1], &hd)));
  shared_ptr<Mat> output;
  graph->Process(shared_ptr<Object>(new AddOp(hd, bd_, &output)));

  graph->Forward();

  prev_hiddens_ = hidden;
  prev_cells_ = cell;

  return output;
}

void lstm::GetParameters(vector<shared_ptr<Mat>> &params)
{
  for (size_t i = 0; i < hidden_sizes_.size(); ++i)
  {
    params.emplace_back(wix_[i]);
    params.emplace_back(wih_[i]);
    params.emplace_back(bi_[i]);
    params.emplace_back(wfx_[i]);
    params.emplace_back(wfh_[i]);
    params.emplace_back(bf_[i]);
    params.emplace_back(wox_[i]);
    params.emplace_back(woh_[i]);
    params.emplace_back(bo_[i]);
    params.emplace_back(wcx_[i]);
    params.emplace_back(wch_[i]);
    params.emplace_back(bc_[i]);
  }
  params.emplace_back(whd_);
  params.emplace_back(bd_);
  params.emplace_back(wil_);
}
