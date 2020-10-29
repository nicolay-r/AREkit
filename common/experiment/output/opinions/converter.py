from arekit.common.experiment.extract.opinions import compose_opinion_collection
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.output.base import BaseOutput

from arekit.common.model.labeling.single import SingleLabelsHelper

from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


class OutputToOpinionCollectionsConverter(object):

    @staticmethod
    def iter_opinion_collections(output_filepath,
                                 opinions_reader,
                                 samples_reader,
                                 experiment,
                                 label_calculation_mode,
                                 output,
                                 keep_news_ids_from_samples_reader,
                                 keep_ids_from_samples_reader):
        assert(isinstance(output_filepath, unicode))
        assert(isinstance(opinions_reader, InputOpinionReader))
        assert(isinstance(samples_reader, InputSampleReader))
        assert(isinstance(label_calculation_mode, unicode))
        assert(isinstance(output, BaseOutput))
        assert(isinstance(keep_news_ids_from_samples_reader, bool))
        assert(isinstance(keep_ids_from_samples_reader, bool))

        output.init_from_tsv(filepath=output_filepath,
                             read_header=True)

        if keep_news_ids_from_samples_reader:
            news_ids_values = list(samples_reader.iter_news_ids())
            output.insert_news_ids_values(news_ids_values)

        if keep_ids_from_samples_reader:
            assert(len(output) == samples_reader.rows_count())
            ids_values = samples_reader.extract_ids()
            output.insert_ids_values(ids_values)

        labels_helper = SingleLabelsHelper(label_scaler=experiment.DataIO.LabelsScaler)

        for news_id in output.iter_news_ids():

            collection = experiment.OpinionOperations.create_opinion_collection()
            assert(isinstance(collection, OpinionCollection))

            linked_iter = output.iter_linked_opinions(news_id=news_id,
                                                      opinions_reader=opinions_reader)

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
