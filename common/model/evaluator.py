from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment.opinions import compose_opinion_collection
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.model.eval.opinion_based import OpinionBasedModelEvaluator


# TODO. Use bert-like evaluator.
# TODO. This should be removed (model does not have evaluation now, only prediction).
# TODO. This should be removed (model does not have evaluation now, only prediction).
# TODO. This should be removed (model does not have evaluation now, only prediction).
class CustomOpinionBasedModelEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, model):
        super(CustomOpinionBasedModelEvaluator, self).__init__(evaluator=evaluator)
        self.__model = model

    def before_evaluation(self, data_type, doc_ids, epoch_index):

        doc_ids_set = set(self.__model.IO.iter_doc_ids_to_compare(doc_ids))

        collections_iter = self.__iter_converted_to_opinion_collections(
            # TODO. This should be based on tsv.
            collection=self.__model.get_text_opinions_collection(data_type),
            create_collection_func=lambda: self.__model.IO.create_opinion_collection(),
            labels_helper=self.__model.LabelsHelper,
            text_opinion_helper=self.__model.get_text_opinion_helper(data_type),
            label_calc_mode=self.__model.Config.TextOpinionLabelCalculationMode)

        used_doc_ids = []
        for doc_opinions, doc_id in collections_iter:

            if doc_id not in doc_ids_set:
                continue

            filepath = self.__model.IO.create_result_opinion_collection_filepath(
                data_type=data_type,
                doc_id=doc_id,
                epoch_index=epoch_index)

            self.__model.IO.DataIO.OpinionFormatter.save_to_file(
                collection=doc_opinions,
                filepath=filepath)

            used_doc_ids.append(doc_id)

        print "Data Type: {}".format(data_type)
        print "Collections saved: {}".format(len(used_doc_ids))
        print "News list: [{lst}]".format(lst=", ".join([str(i) for i in used_doc_ids]))

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        return self.__model.IO.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)

    @staticmethod
    def __iter_converted_to_opinion_collections(collection,
                                                create_collection_func,
                                                text_opinion_helper,
                                                labels_helper,
                                                label_calc_mode):
        assert(isinstance(collection, LinkedTextOpinionCollection))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))
        assert(callable(create_collection_func))
        assert(isinstance(label_calc_mode, unicode))

        for news_id in collection.get_unique_news_ids():

            collection = compose_opinion_collection(
                create_collection_func=create_collection_func,
                linked_data_iter=collection.iter_wrapped_linked_text_opinions(news_id=news_id),
                labels_helper=labels_helper,
                to_opinion_func=text_opinion_helper.to_opinion,
                label_calc_mode=label_calc_mode)

            yield collection, news_id
