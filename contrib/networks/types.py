from enum import Enum


class ModelNames(Enum):

    CNN = u'cnn'
    AttEndsCNN = u'att-cnn'
    AttEndsAndFramesCNN = u'att-ef-cnn'
    AttSynonymEndsCNN = u'att-se-cnn'
    AttSynonymEndsPCNN = u'att-se-pcnn'
    AttSynonymEndsBiLSTM = u'att-se-bilstm'
    AttSynonymEndsAndFramesCNN = u'att-sef-cnn'
    AttSynonymEndsAndFramesPCNN = u'att-sef-pcnn'
    AttSynonymEndsAndFramesBiLSTM = u'att-sef-bilstm'
    AttEndsAndFramesPCNN = u'att-ef-pcnn'
    AttEndsAndFramesBiLSTM = u'att-ef-bilstm'
    AttEndsPCNN = u'att-pcnn'
    AttFramesCNN = u'att-frames-cnn'
    AttFramesPCNN = u'att-frames-pcnn'
    SelfAttentionBiLSTM = u'self-att-bilstm'
    BiLSTM = u'bilstm'
    IANFrames = u'ian'
    IANEnds = u'ian-ends'
    IANEndsAndFrames = u'ian-ef'
    IANSynonymEnds = u'ian-se'
    IANSynonymEndsAndFrames = u'ian-sef'
    PCNN = u'pcnn'
    LSTM = u'rnn'
    RCNN = u'rcnn'
    RCNNAttPZhou = u'rcnn-att-p-zhou'
    RCNNAttZYang = u'rcnn-att-z-yang'
    AttFramesBiLSTM = u'att-frames-bilstm'
    AttSelfZYangBiLSTM = u'att-bilstm-z-yang'
    AttSelfPZhouBiLSTM = u'att-bilstm'


class ModelNamesService(object):

    __names = dict([(item.value, item) for item in ModelNames])

    @staticmethod
    def get_type_by_name(name):
        return ModelNamesService.__names[name]

    @staticmethod
    def iter_supported_names():
        return iter(ModelNamesService.__names.keys())

