class DataIO(object):

    @property
    def NeutralAnnontator(self):
        raise NotImplementedError()

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

    def get_doc_stat_filepath(self):
        raise NotImplementedError()
