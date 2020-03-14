
class BaseAnnotator(object):
    """
    Performs neutral annotation for different data_type.
    """

    def create(self, data_type):
        raise NotImplementedError()

    def get_opin_filepath(self, doc_id, data_type, output_dir):
        raise NotImplementedError()


