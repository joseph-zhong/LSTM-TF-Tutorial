# RNN

Tutorial for RNNs in Tensorflow

- 9 Sept. 2017
- Tutorial Link: https://www.tensorflow.org/tutorials/recurrent
- Tutorial Code:
  https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb

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

## Truncated Backpropagation

Because outputs of RNNs depend on arbitrarily distant inputs, backpropagation
computation can be difficult. We can alleviate this by performing "Truncated
Backpropagation", where we unroll the network, and compute a fixed number of
steps, and then train on this finite approximation. 

### Code Example

```python
# TF Placeholder for the inputs.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

# Initialize LSTM and initial state of memory.
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
initial_state = state = tf.zeros([batch_size, lstm.state_size])

# Perform finite number of update steps to compute output and state.
for i in xrange(num_steps):
  output, state = lstm(words[:, i], state)
  ...
```

#### Training Loop

```python
state = initial_state.eval()
total_loss = 0.0
for curr_word_batch in dataset:
  state, curr_loss = session.run([final_state, loss], 
                                 feed_dict={
                                   initial_state: state,
                                   words: curr_word_batch
                                 })
  total_loss += curr_loss
```

## Loss Function

The goal is to minimize the average negative log probability on target words

```
loss = -1/n * sum(ln(p_target(i)) for i in xrange(n))
```

### Perplexity Metric

Average per-word perplexity measures how well a probability distribution
predicts a sample, where a lower perplexity indicates a representative
distribution.

- Fundamentally, the perplexity of a discrete probability distribution $p$ is
  defined as a function of the entropy (in bits) of the distribution and x
  events
  ```latex
  2^{H(p)} = 2^{-sum_x p(x) lg(p(x)) }
  ```
- Wikipedia: https://en.wikipedia.org/wiki/Perplexity

## LSTMs

To add additional parameters for a more expessive model, we can simply add
additional layers.

### Initialization

- Main modules:
  [`tf.contrib.rnn`](https://www.tensorflow.org/api_guides/python/contrib.rnn)

```python
class Lstm(object):
  def __init__(self, size, num_layers):
    self.lstm = LstmCell._lstm(size, num_layers) 

  @staticmethod
  def _lstm(size, num_layers):
    # REVIEW josephz: Does list multiplication syntax makes sense here? It seems
    # cleaner than iteration, although a generator may work as well.
    cells = tuple(tf.contrib.rnn.BasicLSTMCell(num_units=size)) * num_layers
    return tf.contrib.rnn.MultiRNNCell(cells=cells)
```

### Forward Pass

```python
# Config.
vocab_size = 10000
hidden_size = 200
size = [vocab_size, hidden_size]
num_layers = 2

# Initialize model and initial state.
model = Lstm(size, num_layers)
initial_state = state = model.lstm.zero_state(batch_size, tf.float32)

# Update step.
for i in xrange(num_steps):
  output, state = model.lstm(words[:, i], state)
  ...

final_state = state
```

