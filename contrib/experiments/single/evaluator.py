from arekit.common.opinions.collection import OpinionCollection
from arekit.networks.eval.opinion_based import OpinionBasedModelEvaluator


class CustomOpinionBasedModelEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, model):
        super(CustomOpinionBasedModelEvaluator, self).__init__(evaluator=evaluator)
        self.__model = model

    def before_evaluation(self, data_type, doc_ids, epoch_index):

        doc_ids = list(self.__model.IO.iter_doc_ids_to_compare(doc_ids))

        for collection, doc_id in doc_ids:
            assert(isinstance(collection, OpinionCollection))

            filepath = self.__model.IO.create_result_opinion_collection_filepath(
                data_type=data_type,
                doc_id=doc_id,
                epoch_index=epoch_index)

            self.__model.IO.DataIO.OpinionFormatter.save_to_file(
                collection=collection,
                filepath=filepath)

        print "Data Type: {}".format(data_type)
        print "Collections saved: {}".format(len(doc_ids))
        print "News list: [{lst}]".format(lst=", ".join(doc_ids))

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        return self.__model.IO.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)

