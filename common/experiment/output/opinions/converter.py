from arekit.common.experiment.extract.opinions import fill_opinion_collection
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.output.base import BaseOutput
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion


class OutputToOpinionCollectionsConverter(object):

    @staticmethod
    def iter_opinion_collections(output_filepath,
                                 # TODO. This is a parameter of an output reader (out)
                                 read_header_in_output,
                                 opinions_reader,
                                 labels_scaler,
                                 keep_doc_id_func,
                                 create_opinion_collection_func,
                                 label_calculation_mode,
                                 output,
                                 # TODO. This is a parameter of an output reader (out)
                                 samples_reader=None):
        assert(callable(keep_doc_id_func))
        # TODO. This is a parameter of an output reader (out)
        assert(isinstance(read_header_in_output, bool))
        assert(isinstance(labels_scaler, BaseLabelScaler))
        assert(isinstance(output_filepath, unicode))
        assert(isinstance(opinions_reader, InputOpinionReader))
        assert(callable(create_opinion_collection_func))
        assert(isinstance(label_calculation_mode, unicode))
        assert(isinstance(output, BaseOutput))
        assert(isinstance(samples_reader, InputSampleReader) or samples_reader is None)

        output.init_from_tsv(filepath=output_filepath,
                             read_header=read_header_in_output)

        labels_helper = SingleLabelsHelper(labels_scaler)

        # This is necessary in case when the latter
        # was not provided in an output file.
        # So we obtain such ids from samples.
        # TODO. Move this into specific nested class from BaseOutput.
        # TODO. Named as BertMultiBaseOutput.
        if samples_reader is not None:
            # Exporting such information from samples.
            row_ids = samples_reader.extract_ids()
            news_ids = samples_reader.iter_news_ids()
            print len(samples_reader._df)
            print len(row_ids)
            print len(news_ids)
            print len(output)
            assert(len(row_ids) == len(news_ids) == len(output))
            # Providing the latter into output.
            output.insert_ids_values(row_ids)
            output.insert_news_ids_values(news_ids)

        for news_id in output.iter_news_ids():

            if not keep_doc_id_func(news_id):
                continue

            collection = create_opinion_collection_func()

            linked_iter = output.iter_linked_opinions(news_id=news_id,
                                                      opinions_reader=opinions_reader)

            fill_opinion_collection(
                collection=collection,
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
