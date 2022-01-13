from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.scalers.two import TwoLabelScaler
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from examples.network.embedding import RusvectoresEmbedding


class Common:

    __tags = {
        None: None,
        RuAttitudesVersions.V12: u'ra12',
        RuAttitudesVersions.V20Base: u'ra20b',
        RuAttitudesVersions.V20BaseNeut: u'ra20bn',
        RuAttitudesVersions.V20Large: u'ra20l',
        RuAttitudesVersions.V20LargeNeut: u'ra20ln'
    }

    @staticmethod
    def __create_folding_type_prefix(folding_type):
        assert(isinstance(folding_type,  FoldingType))
        if folding_type == FoldingType.Fixed:
            return u'fx'
        elif folding_type == FoldingType.CrossValidation:
            return u'cv'
        else:
            raise NotImplementedError(u"Folding type `{}` was not declared".format(folding_type))

    @staticmethod
    def __create_input_type_prefix(input_type):
        assert(isinstance(input_type, ModelInputType))
        if input_type == ModelInputType.SingleInstance:
            return u'ctx'
        elif input_type == ModelInputType.MultiInstanceMaxPooling:
            return u'mi-mp'
        elif input_type == ModelInputType.MultiInstanceWithSelfAttention:
            return u'mi-sa'
        else:
            raise NotImplementedError(u"Input type `{}` was not declared".format(input_type))

    @staticmethod
    def create_full_model_name(model_name, input_type):
        assert(isinstance(model_name, ModelNames))
        return u'_'.join([Common.__create_folding_type_prefix(FoldingType.Fixed),
                          Common.__create_input_type_prefix(input_type),
                          model_name.value])

    @staticmethod
    def create_exp_name_suffix(use_balancing, terms_per_context, dist_in_terms_between_att_ends):
        """ Provides an external parameters that assumes to be synchronized both
            by serialization and training experiment stages.
        """
        assert(isinstance(use_balancing, bool))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(dist_in_terms_between_att_ends, int) or dist_in_terms_between_att_ends is None)

        # You may provide your own parameters out there
        params = [
            u"balanced" if use_balancing else u"nobalance",
            u"tpc{}".format(terms_per_context)
        ]

        if dist_in_terms_between_att_ends is not None:
            params.append(u"dbe{}".format(dist_in_terms_between_att_ends))

        return u'-'.join(params)

    @staticmethod
    def create_labels_scaler(labels_count):
        assert(isinstance(labels_count, int))

        if labels_count == 2:
            return TwoLabelScaler()
        if labels_count == 3:
            return ThreeLabelScaler()

        raise NotImplementedError("Not supported")

    @staticmethod
    def load_rusvectores_embedding(filepath, stemmer):
        embedding = RusvectoresEmbedding.from_word2vec_format(filepath=filepath, binary=True)
        embedding.set_stemmer(stemmer)
        return embedding

