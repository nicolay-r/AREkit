import logging

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.common import create_text_parser
from arekit.contrib.experiment_rusentrel.exp_ds.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_ds.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.experiment_rusentrel.exp_joined.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_joined.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_sl.documents import RuSentrelDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_sl.factory import OptionalSynonymsProvider
from arekit.contrib.experiment_rusentrel.exp_sl.opinions import RuSentrelOpinionOperations
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_rusentrel_with_ruattitudes_expriment(exp_ctx, exp_io, folding_type, ra_doc_id_func,
                                                ruattitudes_version, rusentrel_version, load_docs,
                                                ppl_items):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Original Paper (RuAttitudes-1.0): https://www.aclweb.org/anthology/R19-1118/
    """
    assert(isinstance(ruattitudes_version, RuAttitudesVersions))
    assert(isinstance(rusentrel_version, RuSentRelVersions))
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(exp_io, BaseIOUtils))
    assert(isinstance(ppl_items, list) or ppl_items is None)
    assert(callable(ra_doc_id_func))

    optional_data = OptnionalDataProvider(exp_ctx=exp_ctx,
                                          ruattitudes_version=ruattitudes_version,
                                          rusentrel_version=rusentrel_version,
                                          load_docs=load_docs,
                                          ra_doc_id_func=ra_doc_id_func,
                                          ppl_items=ppl_items)

    # init text parser.
    # TODO. Limitation, depending on document, entities parser may vary.
    text_parser = create_text_parser(
        exp_ctx=exp_ctx,
        entities_parser=BratTextEntitiesParser(),
        value_to_group_id_func=optional_data.get_synonyms().get_synonym_group_index,
        ppl_items=ppl_items)

    # init documents.
    rusentrel_doc = RuSentrelDocumentOperations(exp_ctx=exp_ctx,
                                                version=rusentrel_version,
                                                get_synonyms_func=optional_data.get_synonyms,
                                                text_parser=text_parser)

    # Init opinions
    rusentrel_op = RuSentrelOpinionOperations(exp_ctx=exp_ctx,
                                              version=rusentrel_version,
                                              exp_io=exp_io,
                                              get_synonyms_func=optional_data.get_synonyms)

    all_rusentrel_doc_ids = RuSentRelIOUtils.iter_collection_indices(rusentrel_version)

    # Init experiment doc_ops and opin_ops
    doc_ops = RuSentrelWithRuAttitudesDocumentOperations(
        rusentrel_doc_ids=set(all_rusentrel_doc_ids),
        rusentrel_doc=rusentrel_doc,
        get_ruattitudes_doc=optional_data.get_or_load_ruattitudes_doc_ops,
        text_parser=text_parser)

    opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
        rusentrel_op=rusentrel_op,
        get_ruattitudes_op=optional_data.get_or_load_ruattitudes_opin_ops,
        is_rusentrel_doc=lambda doc_id: exp_ctx.DataFolding.contains_doc_id(doc_id))

    return BaseExperiment(exp_ctx=exp_ctx, exp_io=exp_io,
                          opin_ops=opin_ops, doc_ops=doc_ops)


class OptnionalDataProvider(object):

    def __init__(self, exp_ctx, ruattitudes_version, rusentrel_version, load_docs, ra_doc_id_func, ppl_items):
        assert(isinstance(exp_ctx, ExperimentContext))
        assert(isinstance(ppl_items, list) or ppl_items is None)
        self.__exp_ctx = exp_ctx
        self.__load_docs = load_docs
        self.__ruattitudes_version = ruattitudes_version
        self.__synonyms_provider = OptionalSynonymsProvider(version=rusentrel_version)
        self.__ra_doc_id_func = ra_doc_id_func
        self.__ruattitudes_op = None
        self.__ruattitudes_doc = None
        self.__ppl_items = None

    def get_synonyms(self):
        return self.__synonyms_provider.get_or_load_synonyms_collection()

    def get_or_load_ruattitudes_opin_ops(self):
        if self.__ruattitudes_op is None:
            if self.__ru_attitudes is None:
                self.__init_ruattitudes_and_doc_ops()
            self.__ruattitudes_op = RuAttitudesOpinionOperations(ru_attitudes=self.__ru_attitudes)
        return self.__ruattitudes_op

    def get_or_load_ruattitudes_doc_ops(self):
        if self.__ruattitudes_doc is None:
            self.__init_ruattitudes_and_doc_ops()
        return self.__ruattitudes_doc

    def __init_ruattitudes_and_doc_ops(self):
        """ this operation may take a while since it assumes to
            load ru_attitudes collection in memory.
        """
        # Loading ru_attitudes in memory
        ru_attitudes = read_ruattitudes_in_memory(version=self.__ruattitudes_version,
                                                  doc_id_func=self.__ra_doc_id_func,
                                                  keep_doc_ids_only=not self.__load_docs)

        text_parser = create_text_parser(
            exp_ctx=self.__exp_ctx,
            entities_parser=RuAttitudesTextEntitiesParser(),
            value_to_group_id_func=self.__synonyms_provider.get_or_load_synonyms_collection().get_synonym_group_index,
            ppl_items=self.__ppl_items)

        # Completing initialization.
        self.__ruattitudes_doc = RuAttitudesDocumentOperations(
            exp_ctx=self.__exp_ctx,
            ru_attitudes=ru_attitudes,
            text_parser=text_parser)
        self.__ru_attitudes = ru_attitudes