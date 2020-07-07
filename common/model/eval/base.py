# TODO. Model evaluator is weird, because we evaluate model in experiment.
# TODO. Correct -- BaseModelPredictor.
class BaseModelEvaluator(object):

    # TODO. Model should not have such opportunity
    # TODO. As the latter has access to samples only.
    # TODO. Predict -- OK (as an alternative)
    def evaluate(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()
