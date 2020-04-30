from arekit.contrib.bert.formatters.opinion import OpinionsFormatter

from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.experiment.data_type import DataType

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

            text_opinions = extract_text_opinions(
                experiment=experiment,
                data_type=data_type,
                terms_per_context=50)

            opnion_formatter = OpinionsFormatter(data_type=data_type)
            opnion_formatter.format(text_opinions=text_opinions)
            opnion_formatter.to_tsv_by_experiment(experiment=experiment)

            sampler = BertEncoder.__create_formatter(data_type=data_type, formatter_type=sample_formatter)
            sampler.to_samples(text_opinions=text_opinions)
            sampler.to_tsv_by_experiment(experiment=experiment)

    @staticmethod
    def __create_formatter(data_type, formatter_type):
        assert(isinstance(formatter_type, unicode))

        if formatter_type == SampleFormatters.CLASSIF_M:
            return BaseSampleFormatter(data_type=data_type,
                                       label_provider=MultipleLabelProvider(),
                                       text_provider=SingleTextProvider())
        if formatter_type == SampleFormatters.CLASSIF_B:
            return BaseSampleFormatter(data_type=data_type,
                                       label_provider=BinaryLabelProvider(),
                                       text_provider=SingleTextProvider())
        if formatter_type == SampleFormatters.NLI_M:
            return NliMultipleSampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.QA_M:
            return QaMultipleSampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.NLI_B:
            return NliBinarySampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.QA_B:
            return QaBinarySampleFormatter(data_type=data_type)

        return None


