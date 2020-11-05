from os.path import join
from arekit.common.experiment.io_utils import BaseIOUtils


class BertIOUtils(BaseIOUtils):

    def get_target_dir(self):
        """ Provides a main directory for input

            NOTE:
            We consider to save serialized results into model dir,
            rather than experiment dir in a base implementation,
            as model affects on text_b, entities representation, etc.
        """
        return join(super(BertIOUtils, self).get_target_dir(),
                    self._experiment.DataIO.ModelIO.get_model_name())

