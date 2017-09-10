# LSTM 

Tutorial for LSTMs in Tensorflow

- 9 Sept. 2017
- Tutorial Link: https://www.tensorflow.org/tutorials/recurrent

## Introduction

- Core Idea: Sequential Neural Network pipeline of multiple components 
  - Memory State
  - Forget Gate
  - Update Gate
  - ...
- The memory state is initialized with zero vector.
- Update gate updates this memory state with each reading of the input sequence
  - Utilizes a `tanh` activation to add or subtract.
- Forget gate "forgets" or masks out information from the memory over time. 
  - Utilizes a `sigmoid` activation to decide to forget or retain.




