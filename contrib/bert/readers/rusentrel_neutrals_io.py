from os.path import join
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir


class RuSentRelNeutralIOUtils(object):

    @staticmethod
    def get_rusentrel_neutral_opin_filepath(doc_id, is_train, model_name=u"universal"):
        assert(isinstance(doc_id, int))
        assert(isinstance(is_train, bool))
        assert(isinstance(model_name, unicode))
        root = get_path_of_subfolder_in_experiments_dir(model_name)
        return join(root, u"art{}.neut.{}.txt".format(doc_id, u'train' if is_train else u'test'))
