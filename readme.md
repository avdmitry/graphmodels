### Graph Models

Ultimate goal is to implement reference and optimized (blas, cuda, etc.) algorithms on C++ for three major class of Machine learning\Deep learning\AI Graph Models:
- *Feed-forward Neural Networks* including *Convolutional Neural Network* model.
- *Reccurent Neural Networks* including *Long Short-Term Memory* model.
- *Probabilistic Graphical Models*.
- *Spiking Neural Networks*.

#### Algorithms
The following models are implemented:
- Deep Recurrent Neural Networks: vanilla **RNN** and Long Short-Term Memory **LSTM**.

  - Character generation example: [text_gen](apps/text_gen.cc)

- Reinforcement: Q-Learning with function approximation by Deep Neural Network.

  - PuckWorld example: [reinforce](apps/reinforce.cc)

- Deep Feed-forward Neural Networks: fully-connected and Convolutional Neural Network **CNN**.

  - Fully-connected example (~98% accuracy in about 1 min on i7 CPU): [mnist-fc](apps/mnist-fc.cc)
  - Convolutional example (~99% accuracy in about 3 min on i7 CPU): [mnist-conv](apps/mnist-conv.cc)
  - Imagenet classification (Tiny model trained only ~4 days, ~100 hours on 3Gb GeForce GTX 780. Center crop accuracy, top-1: 47.7%, top-5: 72.9%): [imagenet_tiny](apps/imagenet_tiny.cc)
 - Evaluating fully-connected model as Spiking neural network (97.5% accuracy): [mnist-fc-spike_eval](apps/mnist-fc-spike_eval.cc)

#### References
1. Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. Gradient-based learning applied to document recognition, 1998.
2. S. Hochreiter, J. Schmidhuber. Long Short-Term Memory, 1997.
3. A. Krizhevsky, I. Sutskever, G.E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, 2012.
4. G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, R.R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors, 2012.
5. V. Nair, G.E. Hinton. Rectified linear units improve restricted boltzmann machines, 2010.
6. S. Ioffe, C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015.
7. P.U. Diehl, D. Neil, J. Binas, M. Cook, S.C. Liu, M. Pfeiffer. Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing, 2015.

##### Other neural network libraries with similar code (reference implementation, C++, etc.)
- [Andrej Karpathy](https://github.com/karpathy) javascript libraries: [recurrentjs](https://github.com/karpathy/recurrentjs), [reinforcejs](https://github.com/karpathy/reinforcejs), [convnetjs](https://github.com/karpathy/convnetjs)

- [Nitish Srivastava](https://github.com/nitishsrivastava) [convnet with cuda-convnet2](https://github.com/TorontoDeepLearning/convnet)

- [BVLC](http://bvlc.eecs.berkeley.edu/) [caffe](https://github.com/BVLC/caffe/)

#### License
[MIT](license.txt)

