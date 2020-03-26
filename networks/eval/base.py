class BaseModelEvaluator(object):

    def evaluate(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()
