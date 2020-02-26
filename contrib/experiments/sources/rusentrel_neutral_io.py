from os.path import join
from arekit.contrib.experiments import utils
from arekit.contrib.experiments.io_utils_base import IOUtilsBase


# TODO. Remove this class
class RuSentRelNeutralIOUtils(object):

    # TODO. Move to rusentrel_neutral_annot_io.py
    @staticmethod
    def get_rusentrel_neutral_opin_filepath(doc_id, is_train, data_io, model_name=u"universal"):
        assert(isinstance(doc_id, int))
        assert(isinstance(is_train, bool))
        assert(isinstance(model_name, unicode))
        assert(isinstance(data_io, IOUtilsBase))

        root = utils.get_path_of_subfolder_in_experiments_dir(subfolder_name=model_name,
                                                              data_io=data_io)
        return join(root, u"art{}.neut.{}.txt".format(doc_id, u'train' if is_train else u'test'))

