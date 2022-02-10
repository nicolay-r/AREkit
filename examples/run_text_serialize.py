import argparse

from examples.input import EXAMPLES
from examples.network.args.common import RusVectoresEmbeddingFilepathArg, TermsPerContextArg, StemmerArg, InputTextArg
from examples.network.args.serialize import EntityFormatterTypesArg
from examples.pipelines.serialize import run_data_serialization_pipeline

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    InputTextArg.add_argument(parser, default=EXAMPLES["simple"][0])
    RusVectoresEmbeddingFilepathArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    EntityFormatterTypesArg.add_argument(parser)
    StemmerArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    input_text = InputTextArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    embedding_filepath = RusVectoresEmbeddingFilepathArg.read_argument(args)
    entity_fmt = EntityFormatterTypesArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)

    run_data_serialization_pipeline(sentences_text_list=[input_text],
                                    terms_per_context=terms_per_context,
                                    embedding_path=embedding_filepath,
                                    entity_fmt_type=entity_fmt,
                                    stemmer=stemmer)
