from arekit.common.experiment.api.base import BaseExperiment
from examples.network.infer.opin_ops import CustomOpinionOperations
from examples.network.infer.io_utils import CustomIOUtils


class CustomExperiment(BaseExperiment):

    def __init__(self, exp_data, synonyms, doc_ops, labels_formatter, neutral_labels_fmt):

        exp_io = CustomIOUtils(self)

        opin_ops = CustomOpinionOperations(
            labels_formatter=labels_formatter,
            exp_io=exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=neutral_labels_fmt)

        super(CustomExperiment, self).__init__(
            exp_data=exp_data,
            experiment_io=exp_io,
            opin_ops=opin_ops,
            doc_ops=doc_ops,
            name="test",
            extra_name_suffix="test")