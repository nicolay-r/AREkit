import tensorflow as tf

from core.networks.attention.architectures.base import Attention
from core.processing.lemmatization.mystem import MystemWrapper
from core.processing.pos.mystem_wrap import POSMystemWrapper
from core.source.embeddings.base import Embedding
from core.source.embeddings.tokens import TokenEmbedding
from core.networks.attention.configurations.base import AttentionConfig


class LabelCalculationMode:
    FIRST_APPEARED = u'take_first_appeared'
    AVERAGE = u'average'


# TODO. Rename as DefaultConfig
class CommonModelSettings(object):

    GPUMemoryFraction = 0.25

    # private settings
    __test_on_epoch = range(0, 30000, 50)
    __use_class_weights = True
    __dropout_keep = 0.5
    __embedding_dropout_keep = None
    __classes_count = 9
    __keep_tokens = True
    __default_stemmer = MystemWrapper()
    __default_pos_tagger = POSMystemWrapper(__default_stemmer.MystemInstance)
    __terms_per_context = 50
    __bags_per_minibatch = 6
    __bag_size = 1
    __learning_rate = 0.1
    __optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=__learning_rate,
        epsilon=10e-6,
        rho=0.95)

    __word_embedding = None
    __missed_word_embedding = None   # Includes not found words.
    __token_embedding = None
    __frames_embedding = None

    __term_embedding_matrix = None   # Includes embeddings of: words, entities, tokens.
    __class_weights = None
    __use_pos_emb = True
    __use_embedding_dropout = False
    __pos_emb_size = 5
    __dist_emb_size = 5
    __text_opinion_label_calc_mode = LabelCalculationMode.AVERAGE

    __use_attention = True
    __attention_model = None
    __attention_config = AttentionConfig()

    def __init__(self):
        self.__embedding_dropout_keep = 1.0 - 1.0 / self.TermsPerContext

    @property
    def DistanceEmbeddingSize(self):
        return self.__dist_emb_size

    @property
    def TextOpinionLabelCalculationMode(self):
        return self.__text_opinion_label_calc_mode

    @property
    def TermEmbeddingMatrix(self):
        return self.__term_embedding_matrix

    @property
    def TermEmbeddingShape(self):
        return self.__term_embedding_matrix.shape

    @property
    def TotalAmountOfTermsInEmbedding(self):
        """
        Returns vocabulary size -- total amount of words/parsed_news,
        for which embedding has been provided
        """
        return self.TermEmbeddingShape(0)

    def modify_test_on_epochs(self, value):
        assert(isinstance(value, list))
        self.__test_on_epoch = value

    def modify_classes_count(self, value):
        self.__classes_count = value
        self.__class_weights = None

    def modify_learning_rate(self, value):
        assert(isinstance(value, float))
        self.__learning_rate = value

    def modify_use_class_weights(self, value):
        assert(isinstance(value, bool))
        self.__use_class_weights = value

    def modify_use_attention(self, value):
        assert(isinstance(value, bool))
        self.__use_attention = value

    def modify_dropout_keep_prob(self, value):
        assert(isinstance(value, float))
        assert(0 < value <= 1.0)
        self.__dropout_keep = value

    def modify_bags_per_minibatch(self, value):
        assert(isinstance(value, int))
        self.__bags_per_minibatch = value

    def modify_bag_size(self, value):
        assert(isinstance(value, int))
        self.__bag_size = value

    def set_missed_words_embedding(self, embedding):
        assert(isinstance(embedding, Embedding))
        assert(self.__missed_word_embedding is None)
        self.__missed_word_embedding = embedding

    def set_term_embedding(self, embedding_matrix):
        assert(self.__term_embedding_matrix is None)
        assert(self.__attention_model is None)

        self.__term_embedding_matrix = embedding_matrix
        self.__attention_model = Attention(cfg=self.__attention_config,
                                           batch_size=self.BatchSize,
                                           terms_per_context=self.TermsPerContext,
                                           term_embedding_size=self.TermEmbeddingShape[1])

    def set_token_embedding(self, token_embedding):
        assert(isinstance(token_embedding, TokenEmbedding))
        assert(self.__token_embedding is None)
        self.__token_embedding = token_embedding

    def set_class_weights(self, class_weights):
        assert(isinstance(class_weights, list))
        assert(len(class_weights) == self.__classes_count)
        self.__class_weights = class_weights

    def set_word_embedding(self, embedding):
        assert(isinstance(embedding, Embedding))
        assert(self.__word_embedding is None)
        self.__word_embedding = embedding

    def set_frames_embedding(self, embedding):
        assert(isinstance(embedding, Embedding))
        assert(self.__frames_embedding is None)
        self.__frames_embedding = embedding

    @property
    def ClassesCount(self):
        return self.__classes_count

    @property
    def Stemmer(self):
        return self.__default_stemmer

    @property
    def PosTagger(self):
        return self.__default_pos_tagger

    @property
    def ClassWeights(self):
        return self.__class_weights

    @property
    def Optimiser(self):
        return self.__optimiser

    @property
    def TestOnEpochs(self):
        return self.__test_on_epoch

    @property
    def BatchSize(self):
        return self.BagSize * self.BagsPerMinibatch

    @property
    def BagSize(self):
        return self.__bag_size

    @property
    def BagsPerMinibatch(self):
        return self.__bags_per_minibatch

    @property
    def DropoutKeepProb(self):
        return self.__dropout_keep

    @property
    def KeepTokens(self):
        return self.__keep_tokens

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    @property
    def UseClassWeights(self):
        return self.__use_class_weights

    @property
    def WordEmbedding(self):
        return self.__word_embedding

    @property
    def MissedWordEmbedding(self):
        return self.__missed_word_embedding

    @property
    def TokenEmbedding(self):
        return self.__token_embedding

    @property
    def FrameEmbedding(self):
        return self.__frames_embedding

    @property
    def UsePOSEmbedding(self):
        return self.__use_pos_emb

    @property
    def PosEmbeddingSize(self):
        return self.__pos_emb_size

    @property
    def LearningRate(self):
        return self.__learning_rate

    @property
    def Epochs(self):
        return max(self.TestOnEpochs) + 1

    @property
    def EmbeddingDropoutKeepProb(self):
        return self.__embedding_dropout_keep

    @property
    def UseEmbeddingDropout(self):
        return self.__use_embedding_dropout

    @property
    def UseAttention(self):
        return self.__use_attention

    @property
    def AttentionModel(self):
        return self.__attention_model

    @property
    def AttentionConfig(self):
        return self.__attention_config

    def _internal_get_parameters(self):
        base_parameters = [
            ("base:use_class_weights", self.UseClassWeights),
            ("base:dropout (keep prob)", self.DropoutKeepProb),
            ("base:classes_count", self.ClassesCount),
            ("base:keep_tokens", self.KeepTokens),
            ("base:class_weights", self.ClassWeights),
            ("base:default_stemmer",  self.Stemmer),
            ("base:default_pos_tagger", self.PosTagger),
            ("base:terms_per_context", self.TermsPerContext),
            ("base:bags_per_minibatch", self.BagsPerMinibatch),
            ("base:bag_size", self.BagSize),
            ("base:batch_size", self.BatchSize),
            ("base:use_pos_emb", self.UsePOSEmbedding),
            ("base:pos_emb_size", self.PosEmbeddingSize),
            ("base:dist_embedding_size", self.DistanceEmbeddingSize),
            ("base:text_opinion_label_calc_mode", self.TextOpinionLabelCalculationMode),
            ("base:use_embedding_dropout", self.UseEmbeddingDropout),
            ("base:embedding dropout (keep prob)", self.EmbeddingDropoutKeepProb),
            ("base:optimizer", self.Optimiser),
            ("base:learning_rate", self.LearningRate),
            ("base:use_attention", self.__use_attention)
        ]

        if self.UseAttention:
            base_parameters += self.AttentionConfig.get_parameters()

        return base_parameters

    def get_parameters(self):
        return [list(p) for p in zip(*self._internal_get_parameters())]
