### Graph Models

*01 August 2015*

Recently I am interested in reinforcement learning and I also wanted to play with recurent neural networks for some time, after a bunch of image captioning papers were published. For both these models I found nice reference implementation on javascript from [Andrej Karpathy](https://github.com/karpathy): [recurrentjs](https://github.com/karpathy/recurrentjs) & [reinforcejs](https://github.com/karpathy/reinforcejs) and his useful article: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Initial code in general follows the code from two libraries above and its main purpose is to implement reference algorithms on C++ (Code works on CPU without a lot of optimizations and GPU using).

#### Algorithms
The following models are implemented:
- Deep Recurrent Neural Networks: vanilla **RNN** and Long Short-Term Memory (**LSTM**).

  Character generation demo: [text_gen](apps/text_gen.cc)

- Reinforcement: Q-Learning with function approximation with Deep Neural Network.

  PuckWorld demo: [reinforce](apps/reinforce.cc)

#### Notes
Currently there are no documentations and any GUI demo.
Tools output plain costs\rewards, samples etc. to console.

For Convolutional Neural Networks I am using nice library [convnet with cuda-convnet2](https://github.com/TorontoDeepLearning/convnet) from [Nitish Srivastava](https://github.com/nitishsrivastava) and [BVLC](http://bvlc.eecs.berkeley.edu/) [caffe](https://github.com/BVLC/caffe/) as it has a lot of implemented algorithms.
So currently I don't plan to implement Convolutional & Pooling layers, but I think having their reference implementation here will be useful and I'll think about adding them later.

#### License
[MIT](license.txt)

