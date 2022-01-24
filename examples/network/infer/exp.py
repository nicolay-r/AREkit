from arekit.common.experiment.api.base import BaseExperiment
from examples.network.infer.exp_io import InferIOUtils
from examples.network.infer.opin_ops import CustomOpinionOperations


class CustomExperiment(BaseExperiment):

    def __init__(self, exp_ctx, synonyms, doc_ops, labels_formatter, neutral_labels_fmt):

        exp_io = InferIOUtils(self)

        opin_ops = CustomOpinionOperations(
            labels_formatter=labels_formatter,
            exp_io=exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=neutral_labels_fmt)

        super(CustomExperiment, self).__init__(exp_ctx=exp_ctx, experiment_io=exp_io,
                                               opin_ops=opin_ops, doc_ops=doc_ops)
