from arekit.contrib.bert.formatters.opinion import OpinionsFormatter

from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.experiment.data_type import DataType

from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter
from arekit.contrib.bert.formatters.sample.formats import SampleFormatters
from arekit.contrib.bert.formatters.sample.nli_b import NliBSampleFormatter
from arekit.contrib.bert.formatters.sample.nli_m import NliMSampleFormatter
from arekit.contrib.bert.formatters.sample.qa_b import QaBSampleFormatter
from arekit.contrib.bert.formatters.sample.qa_m import QaMSampleFormatter


class BertEncoder(object):

    @staticmethod
    def to_tsv(experiment, sample_formatter):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(sample_formatter, unicode))

        terms_per_context = 50

        for data_type in DataType.iter_supported():
            experiment.DataIO.NeutralAnnotator.create_collection(data_type)

        for data_type in DataType.iter_supported():

            text_opinions = extract_text_opinions(
                experiment=experiment,
                data_type=data_type,
                terms_per_context=terms_per_context)

            #
            # Compose csv file with related opinions (Necessary for evaluation)
            #
            OpinionsFormatter.create_and_save_opinions_to_csv(text_opinions=text_opinions,
                                                              data_type=data_type,
                                                              experiment=experiment)

            sampler = BertEncoder.__create_formatter(data_type=data_type, formatter_type=sample_formatter)
            sampler.to_samples(text_opinions=text_opinions)
            sampler.to_tsv_by_experiment(experiment=experiment)

    @staticmethod
    def __create_formatter(data_type, formatter_type):
        assert(isinstance(formatter_type, unicode))

        if formatter_type == SampleFormatters.DEFAULT:
            return BaseSampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.NLI_M:
            return NliMSampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.QA_M:
            return QaMSampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.NLI_B:
            return NliBSampleFormatter(data_type=data_type)
        if formatter_type == SampleFormatters.QA_B:
            return QaBSampleFormatter(data_type=data_type)

        return None


