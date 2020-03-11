class BaseExperimentsIOUtils(object):
    """
    Related methods implementation should be
    declared outside of an AREkit framework.
    """

    @property
    def NeutralAnnontator(self):
        raise NotImplementedError()

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

    def get_capitals_list(self):
        raise NotImplementedError()

    def get_states_list(self):
        raise NotImplementedError()

    def get_doc_stat_filepath(self):
        raise NotImplementedError()
