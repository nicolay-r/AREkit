from arekit.contrib.experiments.single.helpers.text_opinions import LabeledLinkedTextOpinionCollectionHelper
from arekit.networks.eval.opinion_based import OpinionBasedModelEvaluator


class CustomOpinionBasedModelEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, model):
        super(CustomOpinionBasedModelEvaluator, self).__init__(evaluator=evaluator)
        self.__model = model

    def before_evaluation(self, data_type, doc_ids, epoch_index):

        doc_ids_set = set(self.__model.IO.iter_doc_ids_to_compare(doc_ids))

        helper = self.__model.get_text_opinions_collection_helper(data_type)
        assert(isinstance(helper, LabeledLinkedTextOpinionCollectionHelper))

        collections_iter = helper.iter_converted_to_opinion_collections(
            create_collection_func=lambda: self.__model.IO.create_opinion_collection(),
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
        print "News list: [{lst}]".format(lst=", ".join(used_doc_ids))

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        return self.__model.IO.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)

