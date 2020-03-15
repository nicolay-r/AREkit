class DataIO(object):

    @property
    def Stemmer(self):
        raise NotImplementedError()

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    @property
    def NeutralAnnontator(self):
        raise NotImplementedError()

    @property
    def KeepTokens(self):
        return True

    @property
    def OpinionFormatter(self):
        raise NotImplementedError()

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

    def get_doc_stat_filepath(self):
        raise NotImplementedError()
