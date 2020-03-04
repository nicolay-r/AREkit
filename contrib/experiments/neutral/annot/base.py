
class BaseAnnotator(object):
    """
    Performs neutral annotation for different data_type.
    """

    # TODO. Use data_type instead!
    def create(self, is_train):
        raise NotImplementedError()

    # TODO. Add
    # def get_opin_filepath(doc_id, is_train, experiments_io, model_name=u"universal"):
    #    raise NotImplementedError