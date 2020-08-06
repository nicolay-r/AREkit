# AREnets-0.20.6

Implementation of neural-netwoks, based on Tensorflow for sentiment attitude extraction task.

### Neural Network Models

* **Aspect-based Attentive encoders**:
    - Multilayer Perceptron (MLP)
        [[code]](networks/attention/architectures/mlp.py) /
        [[github:nicolay-r]](https://github.com/nicolay-r/mlp-attention);
* **Self-based Attentive encoders**:
    - P. Zhou et. al.
        [[code]](networks/attention/architectures/self_p_zhou.py) /
        [[github:SeoSangwoo]](https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction);
    - Z. Yang et. al.
        [[code]](networks/attention/architectures/self_z_yang.py) /
        [[github:ilivans]](https://github.com/ilivans/tf-rnn-attention);
* **Single Sentence Based Architectures**:
    - CNN
        [[code]](networks/context/architectures/cnn.py) /
        [[github:roomylee]](https://github.com/roomylee/cnn-relation-extraction);
    - CNN + Aspect-based MLP Attention
        [[code]](networks/context/architectures/base/att_cnn_base.py);
    - PCNN
        [[code]](networks/context/architectures/pcnn.py) /
        [[github:nicolay-r]](https://github.com/nicolay-r/sentiment-pcnn);
    - PCNN + Aspect-based MLP Attention
        [[code]](networks/context/architectures/base/att_pcnn_base.py);
    - RNN (LSTM/GRU/RNN)
        [[code]](networks/context/architectures/rnn.py) /
        [[github:roomylee]](https://github.com/roomylee/rnn-text-classification-tf);
    - IAN (frames based)
        [[code]](networks/context/architectures/ian_frames.py) /
        [[github:lpq29743]](https://github.com/lpq29743/IAN);
    - RCNN (BiLSTM + CNN)
        [[code]](networks/context/architectures/rcnn.py) /
        [[github:roomylee]](https://github.com/roomylee/rcnn-text-classification);
    - RCNN + Self Attention
        [[code]](networks/context/architectures/rcnn_self.py);
    - BiLSTM
        [[code]](networks/context/architectures/bilstm.py) /
        [[github:roomylee]](https://github.com/roomylee/rnn-text-classification-tf);
    - Bi-LSTM + Aspect-based MLP Attention 
        [[code]](networks/context/architectures/base/att_bilstm_base.py)
    - Bi-LSTM + Self Attention
        [[code]](networks/context/architectures/self_att_bilstm.py) /
        [[github:roomylee]](https://github.com/roomylee/self-attentive-emb-tf);
    - RCNN + Self Attention
        [[code]](networks/context/architectures/att_self_rcnn.py);
* **Multi Sentence Based Encoders Architectures**:
    - Self Attentive 
        [[code]](networks/multi/architectures/att_self.py);
    - Max Pooling
        [[code]](networks/multi/architectures/max_pooling.py) /
        [[paper]](https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf);
    - Single MLP
        [[code]](networks/multi/architectures/base/base_single_mlp.py);

## References

TODO.