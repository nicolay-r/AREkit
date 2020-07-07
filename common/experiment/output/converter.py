from arekit.common.experiment.output.base import BaseOutput
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.opinions import compose_opinion_collection

from arekit.common.model.labeling.single import SingleLabelsHelper

from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


class OutputToOpinionCollectionsConverter(object):

    @staticmethod
    def iter_opinion_collections(samples_formatter_func,
                                 experiment,
                                 label_calculation_mode,
                                 output):
        """
        Args:
            samples_formatter_func: func(data_type) -> FormatterType
        """
        assert(callable(samples_formatter_func))
        assert(isinstance(label_calculation_mode, unicode))
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(output, BaseOutput))

        data_type = DataType.Test

        bert_test_samples = samples_formatter_func(data_type)
        bert_test_samples.from_tsv(experiment=experiment)

        output.from_tsv(data_type=data_type,
                        experiment=experiment,
                        ids_values=bert_test_samples.extract_ids())

        bert_test_opinions = BaseOpinionsFormatter(data_type=data_type)
        bert_test_opinions.from_tsv(experiment=experiment)

        assert(len(output) == len(bert_test_samples))

        labels_helper = SingleLabelsHelper(label_scaler=experiment.DataIO.LabelsScaler)

        for news_id in output.iter_news_ids():

            collection = experiment.OpinionOperations.create_opinion_collection()
            assert(isinstance(collection, OpinionCollection))

            linked_iter = output.iter_linked_opinions(news_id=news_id,
                                                      opinions_formatter=bert_test_opinions)

            collection = compose_opinion_collection(
                create_collection_func=experiment.OpinionOperations.create_opinion_collection,
                linked_data_iter=linked_iter,
                labels_helper=labels_helper,
                to_opinion_func=OutputToOpinionCollectionsConverter.__to_label,
                label_calc_mode=label_calculation_mode)

            yield news_id, collection

    @staticmethod
    def __to_label(item, label):
        assert(isinstance(item, Opinion))
        return Opinion(source_value=item.SourceValue,
                       target_value=item.TargetValue,
                       sentiment=label)
