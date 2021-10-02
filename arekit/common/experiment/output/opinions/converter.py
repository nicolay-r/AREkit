from arekit.common.experiment.input.views.opinions import BaseOpinionStorageView
from arekit.common.experiment.output.utils import fill_opinion_collection
from arekit.common.experiment.output.views.base import BaseOutputView
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.opinions.base import Opinion


class OutputToOpinionCollectionsConverter(object):

    # TODO. To output_view. Provide opinions_iter for collection organization only!
    @staticmethod
    def iter_opinion_collections(opinions_view,
                                 # TODO. Remove
                                 labels_scaler,
                                 keep_doc_id_func,
                                 # TODO. Collection???
                                 create_opinion_collection_func,
                                 # TODO. Remove
                                 label_calculation_mode,
                                 # TODO. Remove
                                 supported_labels,
                                 output_formatter):
        assert(callable(keep_doc_id_func))
        # TODO. Only for collection filling (remove)
        assert(isinstance(labels_scaler, BaseLabelScaler))
        assert(isinstance(opinions_view, BaseOpinionStorageView))
        # TODO. Collection???
        assert(callable(create_opinion_collection_func))
        # TODO. Only for collection filling (remove)
        assert(isinstance(label_calculation_mode, LabelCalculationMode))
        # TODO. Only for collection filling (remove)
        assert(isinstance(supported_labels, set) or supported_labels is None)
        assert(isinstance(output_formatter, BaseOutputView))

        labels_helper = SingleLabelsHelper(labels_scaler)

        for news_id in output_formatter.iter_news_ids():

            if not keep_doc_id_func(news_id):
                continue

            # TODO. Collection???
            collection = create_opinion_collection_func()

            linked_iter = output_formatter.iter_linked_opinions(news_id=news_id,
                                                                opinions_view=opinions_view)

            fill_opinion_collection(
                collection=collection,
                linked_data_iter=linked_iter,
                labels_helper=labels_helper,
                to_opinion_func=OutputToOpinionCollectionsConverter.__to_label,
                label_calc_mode=label_calculation_mode,
                supported_labels=supported_labels)

            yield news_id, collection

    @staticmethod
    def __to_label(item, label):
        assert(isinstance(item, Opinion))
        return Opinion(source_value=item.SourceValue,
                       target_value=item.TargetValue,
                       sentiment=label)
