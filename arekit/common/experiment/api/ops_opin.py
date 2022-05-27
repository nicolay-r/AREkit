class OpinionOperations(object):
    """
    Provides operations with opinions and related collections
    """

    def __init__(self):
        pass

    # region annotation

    @property
    def LabelsFormatter(self):
        raise NotImplementedError()

    # endregion

    # region extraction

    # TODO. #248 remove doc_ops (see details in the issue description).
    def iter_annot_collections(self, exp_ctx, doc_ops, data_type):

        collections_it = exp_ctx.Annotator.iter_annotated_collections(
            data_type=data_type, doc_ops=doc_ops, opin_ops=self)

        for doc_id, collection in collections_it:
            yield doc_id, collection

    def iter_opinions_for_extraction(self, doc_id, data_type):
        """ providing opinions for further context-level opinion extraction process.
            in terms of sentiment attitude extraction, this is a general method
            which provides all the possible opinions within a particular document.
        """
        raise NotImplementedError()

    # endregion

    # region evaluation

    def get_etalon_opinion_collection(self, doc_id):
        raise NotImplementedError()

    def get_result_opinion_collection(self, doc_id, data_type, epoch_index):
        raise NotImplementedError()

    # endregion

    # region creation

    def create_opinion_collection(self, opinions):
        raise NotImplementedError("Collection creation does not supported by experiment.")

    # endregion
