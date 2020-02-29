# TODO. Rename as ExperimentsIOUtilsBase
class BaseExperimentsIO(object):
    """
    Related methods implementation should be
    declared outside of an AREkit framework.
    """

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

    def get_capitals_list(self):
        raise NotImplementedError()

    def get_states_list(self):
        raise NotImplementedError()

    def get_rusvectores_news_embedding_filepath(self):
        raise NotImplementedError()

    def get_doc_stat_filepath(self):
        raise NotImplementedError()
