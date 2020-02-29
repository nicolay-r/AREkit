## Neural Networks

### Convolutional Neural Networks

#### CNN

#### Piecewise CNN

### Recurrent Neural Networks (Sequence-Based Text Presentation)

### Attention Architectures

#### IAN

Includes:
* Frame aspect based implementation [[code]](context/architectures/ian_frames.py);
* Attitude ends aspect based implementation;
> NOTE: Experiments with RuSentRel results in an application of base Optimizer instead of 
`tf.train.AdamOptimizer(learning_rate=learning_rate)` oprimizer. The latter stucks training process.

#### Att-BiLSTM

### Training Approaches
    
1. Single Sentence Training

2. Multiple Sentence Training

#### Layers Regularization

We utilize 'L2'-regularization for layers and then combine with the ordinary loss 
([stack-overflow-post](https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow#37143333)):
```
tf.get_variable('a', regularizer=tf.contrib.layers.l2_regularizer(0.001))
loss = ordinary_loss + tf.losses.get_regularization_loss()
```
