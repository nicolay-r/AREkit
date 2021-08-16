from enum import Enum


class ModelNames(Enum):

    CNN = 'cnn'
    AttEndsCNN = 'att-cnn'
    AttEndsAndFramesCNN = 'att-ef-cnn'
    AttSynonymEndsCNN = 'att-se-cnn'
    AttSynonymEndsPCNN = 'att-se-pcnn'
    AttSynonymEndsBiLSTM = 'att-se-bilstm'
    AttSynonymEndsAndFramesCNN = 'att-sef-cnn'
    AttSynonymEndsAndFramesPCNN = 'att-sef-pcnn'
    AttSynonymEndsAndFramesBiLSTM = 'att-sef-bilstm'
    AttEndsAndFramesPCNN = 'att-ef-pcnn'
    AttEndsAndFramesBiLSTM = 'att-ef-bilstm'
    AttEndsPCNN = 'att-pcnn'
    AttFramesCNN = 'att-frames-cnn'
    AttFramesPCNN = 'att-frames-pcnn'
    SelfAttentionBiLSTM = 'self-att-bilstm'
    BiLSTM = 'bilstm'
    IANFrames = 'ian'
    IANEnds = 'ian-ends'
    IANEndsAndFrames = 'ian-ef'
    IANSynonymEnds = 'ian-se'
    IANSynonymEndsAndFrames = 'ian-sef'
    PCNN = 'pcnn'
    LSTM = 'rnn'
    RCNN = 'rcnn'
    RCNNAttPZhou = 'rcnn-att-p-zhou'
    RCNNAttZYang = 'rcnn-att-z-yang'
    AttFramesBiLSTM = 'att-frames-bilstm'
    AttSelfZYangBiLSTM = 'att-bilstm-z-yang'
    AttSelfPZhouBiLSTM = 'att-bilstm'


class ModelNamesService(object):

    __names = dict([(item.value, item) for item in ModelNames])

    @staticmethod
    def get_type_by_name(name):
        return ModelNamesService.__names[name]

    @staticmethod
    def iter_supported_names():
        return iter(list(ModelNamesService.__names.keys()))

