import datetime

import tensorflow as tf
from arekit.common.embeddings.base import Embedding
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper


class DefaultNetworkConfig(object):

    # region private settings

    __gpu_memory_fraction = 0.25
    __use_class_weights = True

    __dropout_keep_prob = 0.5
    __embedding_dropout_keep_prob = 1.0

    __classes_count = None
    __default_pos_tagger = POSMystemWrapper(MystemWrapper().MystemInstance)
    __terms_per_context = 50
    __synonyms_per_context = 3
    __frames_per_context = 5
    __bags_per_minibatch = 6
    __bag_size = 1
    __learning_rate = 0.1

    __default_weight_initializer = tf.random_normal_initializer(mean=0, stddev=1.0)
    __default_bias_initializer = tf.zeros_initializer(dtype=tf.float32)
    __default_regularizer = None

    __optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=__learning_rate,
        epsilon=10e-6,
        rho=0.95)

    __word_embedding = None
    __custom_word_embedding = None   # Includes not found words.
    __token_embedding = None
    __frames_embedding = None

    __term_embedding_matrix = None   # Includes embeddings of: words, entities, tokens.
    __class_weights = None
    __use_pos_emb = True
    __pos_emb_size = 5
    __sent_emb_size = 5
    __dist_emb_size = 5
    __text_opinion_label_calc_mode = LabelCalculationMode.AVERAGE

    __use_entity_types_in_embedding = True              # Affects on result embedding of related entity: entity + type.
    __use_entity_types_as_context_feature = False       # TODO. Context based feature, i.e. declared for all terms

    __l2_reg = 0.0

    # endregion

    def __init__(self):
        self.__default_regularizer = tf.contrib.layers.l2_regularizer(self.L2Reg)

    # region properties

    @property
    def L2Reg(self):
        return self.__l2_reg

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

    @property
    def BiasInitializer(self):
        return self.__default_bias_initializer

    @property
    def WeightInitializer(self):
        return self.__default_weight_initializer

    @property
    def GPUMemoryFraction(self):
        return self.__gpu_memory_fraction

    @property
    def LayerRegularizer(self):
        return self.__default_regularizer

    @property
    def EmbeddingInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    # endregion

    # region public methods

    def modify_use_entity_types_in_embedding(self, value):
        assert(isinstance(value, bool))
        self.__use_entity_types_in_embedding = value

    def modify_use_entity_types_as_context_feature(self, value):
        assert(isinstance(value, bool))
        self.__use_entity_types_as_context_feature = value

    def modify_terms_per_context(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__terms_per_context = value

    def modify_synonyms_per_context(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__synonyms_per_context = value

    def modify_l2_reg(self, value):
        assert(isinstance(value, float))
        self.__l2_reg = value

    def modify_optimizer(self, value):
        self.__optimiser = value

    def modify_weight_initializer(self, value):
        self.__default_weight_initializer = value

    def modify_bias_initializer(self, value):
        self.__default_bias_initializer = value

    def modify_regularizer(self, value):
        self.__default_regularizer = value

    def modify_classes_count(self, value):
        self.__classes_count = value
        self.__class_weights = None

    def modify_learning_rate(self, value):
        assert(isinstance(value, float))
        self.__learning_rate = value

    def modify_use_class_weights(self, value):
        assert(isinstance(value, bool))
        self.__use_class_weights = value

    def modify_dropout_keep_prob(self, value):
        assert(isinstance(value, float))
        assert(0 < value <= 1.0)
        self.__dropout_keep_prob = value

    def modify_embedding_dropout_keep_prob(self, value):
        assert(isinstance(value, float))
        assert(0 < value <= 1.0)
        self.__embedding_dropout_keep_prob = value

    def modify_bags_per_minibatch(self, value):
        assert(isinstance(value, int))
        self.__bags_per_minibatch = value

    def modify_gpu_memory_fraction(self, value):
        assert(isinstance(value, float))
        self.__gpu_memory_fraction = value

    def modify_bag_size(self, value):
        assert(isinstance(value, int))
        self.__bag_size = value

    def set_custom_words_embedding(self, embedding):
        assert(isinstance(embedding, Embedding))
        assert(self.__custom_word_embedding is None)
        self.__custom_word_embedding = embedding

    def set_term_embedding(self, embedding_matrix):
        assert(self.__term_embedding_matrix is None)
        self.__term_embedding_matrix = embedding_matrix

    def set_token_embedding(self, token_embedding):
        assert(isinstance(token_embedding, Embedding))
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

    def notify_initialization_completed(self):
        pass

    # endregion

    # region properties

    @property
    def ClassesCount(self):
        return self.__classes_count

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
        return self.__dropout_keep_prob

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    @property
    def SynonymsPerContext(self):
        return self.__synonyms_per_context

    @property
    def UseClassWeights(self):
        return self.__use_class_weights

    @property
    def WordEmbedding(self):
        return self.__word_embedding

    @property
    def CustomWordEmbedding(self):
        return self.__custom_word_embedding

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
    def SentimentEmbeddingSize(self):
        return self.__sent_emb_size

    @property
    def LearningRate(self):
        return self.__learning_rate

    @property
    def EmbeddingDropoutKeepProb(self):
        return self.__embedding_dropout_keep_prob

    @property
    def FramesPerContext(self):
        return self.__frames_per_context

    @property
    def UseEntityTypeAsContextFeature(self):
        return self.__use_entity_types_as_context_feature

    # endregion

    def _internal_get_parameters(self):
        return [
            ("base:current_time", datetime.datetime.now()),
            ("base:memory_fraction", self.GPUMemoryFraction),
            ("base:use_class_weights", self.UseClassWeights),
            ("base:dropout (keep prob)", self.DropoutKeepProb),
            ("base:classes_count", self.ClassesCount),
            ("base:class_weights", self.ClassWeights),
            ("base:default_pos_tagger", self.PosTagger),
            ("base:terms_per_context", self.TermsPerContext),
            ("base:synonyms_per_context", self.SynonymsPerContext),
            ("base:bags_per_minibatch", self.BagsPerMinibatch),
            ("base:bag_size", self.BagSize),
            ("base:batch_size", self.BatchSize),
            ("base:use_pos_emb", self.UsePOSEmbedding),
            ("base:pos_emb_size", self.PosEmbeddingSize),
            ("base:sentiment_emb_size", self.SentimentEmbeddingSize),
            ("base:dist_embedding_size", self.DistanceEmbeddingSize),
            ("base:text_opinion_label_calc_mode", self.TextOpinionLabelCalculationMode),
            ("base:use_entity_types_as_context_feature", self.UseEntityTypeAsContextFeature),
            ("base:embedding dropout (keep prob)", self.EmbeddingDropoutKeepProb),
            ("base:optimizer", self.Optimiser),
            ("base:learning_rate", self.LearningRate),
            ("base:l2_reg", self.L2Reg),
            ("base:layer_regularizer", self.LayerRegularizer),
            ("base:weight_initializer", self.WeightInitializer),
            ("base:bias_initializer", self.BiasInitializer),
        ]

    def get_parameters(self):
        return [list(p) for p in zip(*self._internal_get_parameters())]
