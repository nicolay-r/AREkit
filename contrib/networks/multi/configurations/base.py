import tensorflow as tf

from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


class BaseMultiInstanceConfig(DefaultNetworkConfig):

    __contexts_per_opinion = 3
    __context_config = None

    def __init__(self, context_config):
        assert(isinstance(context_config, DefaultNetworkConfig))
        super(BaseMultiInstanceConfig, self).__init__()
        self.__context_config = context_config

        self.__context_parameters_fix()

    # region properties

    @property
    def TextOpinionLabelCalculationMode(self):
        return LabelCalculationMode.FIRST_APPEARED

    @property
    def BagsPerMinibatch(self):
        return super(BaseMultiInstanceConfig, self).BagsPerMinibatch

    @property
    def BatchSize(self):
        return self.BagsPerMinibatch

    @property
    def BagSize(self):
        return self.__contexts_per_opinion

    @property
    def ContextConfig(self):
        return self.__context_config

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    @property
    def BaseInitializer(self):
        return tf.random_uniform_initializer(-0.1, 0.1)

    @property
    def ClassesCount(self):
        return self.__context_config.ClassesCount

    @property
    def LearningRate(self):
        return self.__context_config.LearningRate

    @property
    def TermsPerContext(self):
        return self.__context_config.TermsPerContext

    @property
    def EmbeddingDropoutKeepProb(self):
        return self.__context_config.EmbeddingDropoutKeepProb

    # endregion

    def set_pos_count(self, value):
        super(BaseMultiInstanceConfig, self).set_pos_count(value)
        self.__context_config.set_pos_count(value)

    def set_contexts_per_opinion(self, value):
        self.__contexts_per_opinion = value

    def set_term_embedding(self, embedding_matrix):
        super(BaseMultiInstanceConfig, self).set_term_embedding(embedding_matrix)
        self.__context_config.set_term_embedding(embedding_matrix)

    def set_class_weights(self, class_weights):
        super(BaseMultiInstanceConfig, self).set_class_weights(class_weights)
        self.__context_config.set_class_weights(class_weights)

    def modify_bags_per_minibatch(self, value):
        super(BaseMultiInstanceConfig, self).modify_bags_per_minibatch(value)
        self.__context_parameters_fix()

    def modify_classes_count(self, value):
        super(BaseMultiInstanceConfig, self).modify_classes_count(value)
        self.__context_config.modify_classes_count(value)

    def modify_dropout_keep_prob(self, value):
        super(BaseMultiInstanceConfig, self).modify_dropout_keep_prob(value)
        self.__context_config.modify_dropout_keep_prob(value)

    def modify_learning_rate(self, value):
        super(BaseMultiInstanceConfig, self).modify_learning_rate(value)
        self.__context_config.modify_learning_rate(value)

    def modify_use_class_weights(self, value):
        super(BaseMultiInstanceConfig, self).modify_use_class_weights(value)
        self.__context_config.modify_use_class_weights(value)

    def init_config_dependent_parameters(self):
        super(BaseMultiInstanceConfig, self).init_config_dependent_parameters()
        self.__context_config.init_config_dependent_parameters()

    def fix_context_parameters(self):
        self.__context_parameters_fix()

    def __context_parameters_fix(self):
        self.__context_config.modify_bag_size(1)
        self.__context_config.modify_bags_per_minibatch(self.BatchSize)

    def _internal_get_parameters(self):

        parameters = [
            ("mi:Batch (BagsPerMinibatch)", self.BagsPerMinibatch),
            ("mi:BagSize (ContextsPerOpinion)", self.BagSize),
            ("mi:LearningRate", self.LearningRate),
            ("mi:DropoutKeepProb", self.DropoutKeepProb),
            ("mi:RelationLabelCalculationMode", self.TextOpinionLabelCalculationMode),
            ("mi:Optimiser", self.Optimiser)
        ]

        parameters.extend(self.__context_config._internal_get_parameters())

        return parameters
