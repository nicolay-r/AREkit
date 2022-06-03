from arekit.common.opinions.annot.default import DefaultAnnotator
from arekit.contrib.experiment_rusentrel.annot.two_scale import TwoScaleTaskAnnotator


class ExperimentAnnotatorFactory:

    @staticmethod
    def create(labels_count, create_algo, create_empty_collection_func, get_doc_etalon_opins_func):
        assert(isinstance(labels_count, int))
        assert(callable(create_algo))

        if labels_count == 2:
            return TwoScaleTaskAnnotator(create_empty_collection_func=create_empty_collection_func,
                                         get_doc_etalon_opins_func=get_doc_etalon_opins_func)
        else:
            return DefaultAnnotator(create_algo(),
                                    create_empty_collection_func=create_empty_collection_func,
                                    get_doc_etalon_opins_func=get_doc_etalon_opins_func)
