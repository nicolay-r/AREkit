from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider


class NetworkSerializationContext(object):

    def __init__(self, labels_scaler, pos_tagger, frame_roles_label_scaler, frames_connotation_provider):
        self.__label_provider = MultipleLabelProvider(labels_scaler)
        self.__pos_tagger = pos_tagger
        self.__frame_roles_label_scaler = frame_roles_label_scaler
        self.__frames_connotation_provider = frames_connotation_provider

    @property
    def LabelProvider(self):
        return self.__label_provider

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler

    @property
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def PosTagger(self):
        return self.__pos_tagger