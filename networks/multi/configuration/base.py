from core.networks.context.configurations import CommonModelSettings, LabelCalculationMode


class MIMLRESettings(CommonModelSettings):

    __hidden_size = 300
    __contexts_per_opinion = 5
    __context_settings = None

    def __init__(self, context_settings):
        assert(isinstance(context_settings, CommonModelSettings))
        super(MIMLRESettings, self).__init__()
        self.__context_settings = context_settings
        self.__context_parameters_fix()

    def set_contexts_per_opinion(self, value):
        self.__contexts_per_opinion = value

    def set_term_embedding(self, embedding_matrix):
        super(MIMLRESettings, self).set_term_embedding(embedding_matrix)
        self.__context_settings.set_term_embedding(embedding_matrix)

    def set_missed_words_embedding(self, embedding):
        super(MIMLRESettings, self).set_missed_words_embedding(embedding)
        self.__context_settings.set_missed_words_embedding(embedding)

    def set_class_weights(self, class_weights):
        super(MIMLRESettings, self).set_class_weights(class_weights)
        self.__context_settings.set_class_weights(class_weights)

    def modify_bags_per_minibatch(self, value):
        super(MIMLRESettings, self).modify_bags_per_minibatch(value)
        self.__context_parameters_fix()

    def modify_classes_count(self, value):
        super(MIMLRESettings, self).modify_classes_count(value)
        self.__context_settings.modify_classes_count(value)

    def modify_dropout_keep_prob(self, value):
        super(MIMLRESettings, self).modify_dropout_keep_prob(value)
        self.__context_settings.modify_dropout_keep_prob(value)

    def modify_learning_rate(self, value):
        super(MIMLRESettings, self).modify_learning_rate(value)
        self.__context_settings.modify_learning_rate(value)

    def modify_use_class_weights(self, value):
        super(MIMLRESettings, self).modify_use_class_weights(value)
        self.__context_settings.modify_use_class_weights(value)

    def modify_use_attention(self, value):
        super(MIMLRESettings, self).modify_use_attention(value)
        self.__context_settings.modify_use_attention(value)

    def __context_parameters_fix(self):
        self.__context_settings.modify_bag_size(1)
        self.__context_settings.modify_bags_per_minibatch(self.BatchSize)

    @property
    def LearningRate(self):
        return 0.2

    @property
    def DropoutKeepProb(self):
        return 0.85

    @property
    def TextOpinionLabelCalculationMode(self):
        return LabelCalculationMode.FIRST_APPEARED

    @property
    def BagsPerMinibatch(self):
        return super(MIMLRESettings, self).BagsPerMinibatch

    @property
    def BatchSize(self):
        return self.BagsPerMinibatch

    @property
    def BagSize(self):
        return self.__contexts_per_opinion

    @property
    def ContextSettings(self):
        return self.__context_settings

    @property
    def HiddenSize(self):
        return self.__hidden_size

    def _internal_get_parameters(self):

        parameters = [
            ("mi:Batch (BagsPerMinibatch)", self.BagsPerMinibatch),
            ("mi:BagSize (ContextsPerOpinion)", self.BagSize),
            ("mi:HiddenSize", self.HiddenSize),
            ("mi:LearningRate", self.LearningRate),
            ("mi:DropoutKeepProb", self.DropoutKeepProb),
            ("mi:RelationLabelCalculationMode", self.TextOpinionLabelCalculationMode),
            ("mi:Optimiser", self.Optimiser)
        ]

        parameters.extend(self.__context_settings._internal_get_parameters())

        return parameters
