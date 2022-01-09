from arekit.common.labels.scaler.base import BaseLabelScaler


class SentimentLabelScaler(BaseLabelScaler):

    def invert_label(self, label):
        raise NotImplementedError()
