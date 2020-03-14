class DataIO(object):

    # TODO. Other dependecies (properties).
    # TODO. Remove Stemmer from DefaultConfig

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    @property
    def NeutralAnnontator(self):
        raise NotImplementedError()

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

    def get_doc_stat_filepath(self):
        raise NotImplementedError()
