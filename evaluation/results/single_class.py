from core.evaluation.results.base import BaseEvalResult


class SingleClassEvalResult(BaseEvalResult):

    def __init__(self):
        pass

    def calculate(self):
        pass

    def add_document_results(self, doc_id, prec, recall):
        pass
