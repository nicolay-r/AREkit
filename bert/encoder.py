from arekit.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.bert.formatters.sample.formats import SampleFormatters
from arekit.bert.providers.opinions import OpinionProvider

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.synonyms import SynonymsCollection

# TODO. Remove these dependencies.
# TODO. Remove these dependencies.
# TODO. Remove these dependencies.
from arekit.contrib.bert_samplers.base import create_simple_sample_formatter
from arekit.contrib.bert_samplers.nli_b import NliBinarySampleFormatter
from arekit.contrib.bert_samplers.nli_m import NliMultipleSampleFormatter
from arekit.contrib.bert_samplers.qa_b import QaBinarySampleFormatter
from arekit.contrib.bert_samplers.qa_m import QaMultipleSampleFormatter


class BertEncoder(object):

    @staticmethod
    def to_tsv(experiment, sample_formatter):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(sample_formatter, unicode))

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type)

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            opinion_provider = OpinionProvider.from_experiment(experiment=experiment, data_type=data_type)

            opnion_formatter = BertOpinionsFormatter(data_type=data_type)
            opnion_formatter.format(opinion_provider=opinion_provider)
            opnion_formatter.to_tsv_by_experiment(experiment=experiment)

            sampler = BertEncoder.create_formatter(data_type=data_type,
                                                   formatter_type=sample_formatter,
                                                   label_scaler=experiment.DataIO.LabelsScaler,
                                                   synonyms=experiment.DataIO.SynonymsCollection)
            sampler.format(opinion_provider=opinion_provider)
            sampler.to_tsv_by_experiment(experiment=experiment)

    @staticmethod
    def create_formatter(data_type, formatter_type, label_scaler, synonyms):
        assert(isinstance(formatter_type, unicode))
        assert(isinstance(synonyms, SynonymsCollection))

        if formatter_type == SampleFormatters.CLASSIF_M:
            return create_simple_sample_formatter(data_type=data_type,
                                                  label_scaler=label_scaler,
                                                  synonyms=synonyms)
        if formatter_type == SampleFormatters.NLI_M:
            return NliMultipleSampleFormatter(data_type=data_type,
                                              label_scaler=label_scaler,
                                              synonyms=synonyms)
        if formatter_type == SampleFormatters.QA_M:
            return QaMultipleSampleFormatter(data_type=data_type,
                                             label_scaler=label_scaler,
                                             synonyms=synonyms)
        if formatter_type == SampleFormatters.NLI_B:
            return NliBinarySampleFormatter(data_type=data_type,
                                            label_scaler=label_scaler,
                                            synonyms=synonyms)
        if formatter_type == SampleFormatters.QA_B:
            return QaBinarySampleFormatter(data_type=data_type,
                                           label_scaler=label_scaler,
                                           synonyms=synonyms)

        return None


