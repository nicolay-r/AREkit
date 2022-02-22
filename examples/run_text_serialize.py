import argparse

from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from examples.input import EXAMPLES
from examples.network.args import const
from examples.network.args.common import RusVectoresEmbeddingFilepathArg, TermsPerContextArg, StemmerArg, InputTextArg, \
    EntitiesParserArg
from examples.network.args.serialize import EntityFormatterTypesArg
from examples.pipelines.serialize import TextSerializationPipelineItem


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    InputTextArg.add_argument(parser, default=EXAMPLES["no_entities"])
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    EntityFormatterTypesArg.add_argument(parser, default="simple")
    StemmerArg.add_argument(parser, default="mystem")

    # Parsing arguments.
    args = parser.parse_args()

    ppl = BasePipeline([
        TextSerializationPipelineItem(
            terms_per_context=TermsPerContextArg.read_argument(args),
            entities_parser=EntitiesParserArg.read_argument(args),
            embedding_path=RusVectoresEmbeddingFilepathArg.read_argument(args),
            entity_fmt=create_entity_formatter(EntityFormatterTypesArg.read_argument(args)),
            stemmer=StemmerArg.read_argument(args))
    ])

    ppl.run(InputTextArg.read_argument(args))
