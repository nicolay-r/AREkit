from pymystem3 import Mystem

from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.contrib.bert.formatters.str_entity_fmt import RussianEntitiesFormatter
from arekit.contrib.bert.providers.label.multiple import BertMultipleLabelProvider
from arekit.contrib.bert.providers.opinions import OpinionProvider

from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter
from arekit.contrib.bert.formatters.sample.formats import SampleFormatters
from arekit.contrib.bert.formatters.sample.nli_b import NliBinarySampleFormatter
from arekit.contrib.bert.formatters.sample.nli_m import NliMultipleSampleFormatter
from arekit.contrib.bert.formatters.sample.qa_b import QaBinarySampleFormatter
from arekit.contrib.bert.formatters.sample.qa_m import QaMultipleSampleFormatter
from arekit.contrib.bert.providers.text.single import SingleTextProvider
from arekit.processing.pos.mystem_wrap import POSMystemWrapper


class BertEncoder(object):

    @staticmethod
    def to_tsv(experiment, sample_formatter):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(sample_formatter, unicode))

        for data_type in DataType.iter_supported():
            experiment.NeutralAnnotator.create_collection(data_type)

        # TODO. This should be a part of the initial config
        entities_formatter = RussianEntitiesFormatter(
            pos_tagger=POSMystemWrapper(Mystem(entire_input=False)))

        for data_type in DataType.iter_supported():
            opinion_provider = OpinionProvider.from_experiment(experiment=experiment, data_type=data_type)

            opnion_formatter = BertOpinionsFormatter(data_type=data_type)
            opnion_formatter.format(opinion_provider=opinion_provider)
            opnion_formatter.to_tsv_by_experiment(experiment=experiment)

            sampler = BertEncoder.create_formatter(data_type=data_type,
                                                   formatter_type=sample_formatter,
                                                   label_scaler=experiment.DataIO.LabelsScaler,
                                                   entities_formatter=entities_formatter,
                                                   synonyms=experiment.DataIO.SynonymsCollection)
            sampler.format(opinion_provider=opinion_provider)
            sampler.to_tsv_by_experiment(experiment=experiment)

    @staticmethod
    def create_formatter(data_type, formatter_type, label_scaler, entities_formatter, synonyms):
        assert(isinstance(formatter_type, unicode))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(entities_formatter, StringEntitiesFormatter))

        if formatter_type == SampleFormatters.CLASSIF_M:
            return BaseSampleFormatter(
                data_type=data_type,
                label_provider=BertMultipleLabelProvider(label_scaler=label_scaler),
                text_provider=SingleTextProvider(entities_formatter=entities_formatter,
                                                 synonyms=synonyms))
        if formatter_type == SampleFormatters.NLI_M:
            return NliMultipleSampleFormatter(data_type=data_type,
                                              label_scaler=label_scaler,
                                              synonyms=synonyms,
                                              entities_formatter=entities_formatter)
        if formatter_type == SampleFormatters.QA_M:
            return QaMultipleSampleFormatter(data_type=data_type,
                                             label_scaler=label_scaler,
                                             synonyms=synonyms,
                                             entities_formatter=entities_formatter)
        if formatter_type == SampleFormatters.NLI_B:
            return NliBinarySampleFormatter(data_type=data_type,
                                            label_scaler=label_scaler,
                                            synonyms=synonyms,
                                            entities_formatter=entities_formatter)
        if formatter_type == SampleFormatters.QA_B:
            return QaBinarySampleFormatter(data_type=data_type,
                                           label_scaler=label_scaler,
                                           synonyms=synonyms,
                                           entities_formatter=entities_formatter)

        return None


