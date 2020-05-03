from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter

from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider

from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter
from arekit.contrib.bert.formatters.sample.formats import SampleFormatters
from arekit.contrib.bert.formatters.sample.label.binary import BinaryLabelProvider
from arekit.contrib.bert.formatters.sample.label.multiple import MultipleLabelProvider
from arekit.contrib.bert.formatters.sample.nli_b import NliBinarySampleFormatter
from arekit.contrib.bert.formatters.sample.nli_m import NliMultipleSampleFormatter
from arekit.contrib.bert.formatters.sample.qa_b import QaBinarySampleFormatter
from arekit.contrib.bert.formatters.sample.qa_m import QaMultipleSampleFormatter
from arekit.contrib.bert.formatters.sample.text.single import SingleTextProvider


class BertEncoder(object):

    @staticmethod
    def to_tsv(experiment, sample_formatter):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(sample_formatter, unicode))

        for data_type in DataType.iter_supported():
            experiment.DataIO.NeutralAnnotator.create_collection(data_type)

        for data_type in DataType.iter_supported():
            opinion_provider = OpinionProvider.from_experiment(experiment=experiment, data_type=data_type)

            opnion_formatter = BertOpinionsFormatter(data_type=data_type)
            opnion_formatter.format(opinion_provider=opinion_provider)
            opnion_formatter.to_tsv_by_experiment(experiment=experiment)

            sampler = BertEncoder.create_formatter(data_type=data_type,
                                                   formatter_type=sample_formatter,
                                                   supported_labels=experiment.DataIO.LabelsScale.supported_labels())
            sampler.to_samples(opinion_provider=opinion_provider)
            sampler.to_tsv_by_experiment(experiment=experiment)

    @staticmethod
    def create_formatter(data_type, formatter_type, supported_labels):
        assert(isinstance(formatter_type, unicode))
        assert(isinstance(supported_labels, list))

        if formatter_type == SampleFormatters.CLASSIF_M:
            return BaseSampleFormatter(data_type=data_type,
                                       label_provider=MultipleLabelProvider(supported_labels=supported_labels),
                                       text_provider=SingleTextProvider())
        if formatter_type == SampleFormatters.CLASSIF_B:
            return BaseSampleFormatter(data_type=data_type,
                                       label_provider=BinaryLabelProvider(supported_labels=supported_labels),
                                       text_provider=SingleTextProvider())
        if formatter_type == SampleFormatters.NLI_M:
            return NliMultipleSampleFormatter(data_type=data_type,
                                              supported_labels=supported_labels)
        if formatter_type == SampleFormatters.QA_M:
            return QaMultipleSampleFormatter(data_type=data_type,
                                             supported_labels=supported_labels)
        if formatter_type == SampleFormatters.NLI_B:
            return NliBinarySampleFormatter(data_type=data_type,
                                            supported_labels=supported_labels)
        if formatter_type == SampleFormatters.QA_B:
            return QaBinarySampleFormatter(data_type=data_type,
                                           supported_labels=supported_labels)

        return None


