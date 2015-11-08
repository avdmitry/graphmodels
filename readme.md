### Graph Models

Ultimate goal is to implement reference and optimized (blas, cuda, etc.) algorithms on C++ for three major class of Graph Models:
- *Feed-forward Neural Networks* including *Convolutional Neural Network* model.
- *Reccurent Neural Networks* including *Long Short-Term Memory* model.
- *Probabilistic Graphical Models*.

#### Algorithms
The following models are implemented:
- Deep Recurrent Neural Networks: vanilla **RNN** and Long Short-Term Memory **LSTM**.

  - Character generation example: [text_gen](apps/text_gen.cc)

- Reinforcement: Q-Learning with function approximation by Deep Neural Network.

  - PuckWorld example: [reinforce](apps/reinforce.cc)

- Deep Feed-forward Neural Networks: fully-connected and Convolutional Neural Network *CNN*.

  - Fully-connected example (~98% accuracy in about 5 min on i7 CPU): [mnist-fc](apps/mnist-fc.cc)
  - Convolutional example (~99% accuracy in about 3 min on i7 CPU): [mnist-conv](apps/mnist-conv.cc)

#### References

##### Other neural network libraries with similar code (reference implementation, C++, etc.)
- [Andrej Karpathy](https://github.com/karpathy) javascript libraries: [recurrentjs](https://github.com/karpathy/recurrentjs), [reinforcejs](https://github.com/karpathy/reinforcejs), [convnetjs](https://github.com/karpathy/convnetjs)

- [Nitish Srivastava](https://github.com/nitishsrivastava) [convnet with cuda-convnet2](https://github.com/TorontoDeepLearning/convnet)

- [BVLC](http://bvlc.eecs.berkeley.edu/) [caffe](https://github.com/BVLC/caffe/)

#### License
[MIT](license.txt)

