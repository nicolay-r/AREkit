import unittest
from os.path import join, dirname

from arekit.common.data import const
from arekit.common.experiment.pipelines.opinion_collections import output_to_opinion_collections_pipeline
from arekit.common.pipeline.context import PipelineContext


class TestOutputFormatters(unittest.TestCase):

    __current_dir = dirname(__file__)
    __input_samples_filepath = join(__current_dir, "data/sample_train.tsv.gz")

    def test_output_formatter(self):

        storage = None

        # TODO. Complete.
        ppl = output_to_opinion_collections_pipeline(
            opin_ops=None,
            doc_ids_set=None,
            labels_scaler=None,
            iter_opinion_linkages_func=None,
            label_calc_mode=None,
            supported_labels=None)

        pipeline_ctx = PipelineContext({
            "src": set(storage.iter_column_values(column_name=const.DOC_ID))
        })

        # Running pipeline.
        ppl.run(pipeline_ctx)

        # Iterate over the result.
        for _ in pipeline_ctx.provide("src"):
            pass


if __name__ == '__main__':
    unittest.main()
