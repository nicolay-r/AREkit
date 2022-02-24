import os
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.items.base import BasePipelineItem
from examples.brat_backend import BratBackend
from examples.exp.exp_io import InferIOUtils


class BratBackendPipelineItem(BasePipelineItem):

    def __init__(self, obj_color_types, rel_color_types,
                 label_to_rel, brat_url="http://localhost:8001/"):
        self.__brat_be = BratBackend()
        self.__brat_url = brat_url
        self.__obj_color_types = obj_color_types
        self.__rel_color_types = rel_color_types
        self.__label_to_rel = label_to_rel

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, InferIOUtils))

        template = self.__brat_be.to_html(
            result_data_filepath=pipeline_ctx.provide("predict_fp"),
            samples_data_filepath=input_data.create_samples_writer_target(DataType.Test),
            obj_color_types=self.__obj_color_types,
            rel_color_types=self.__rel_color_types,
            label_to_rel=self.__label_to_rel)

        # Setup predicted result writer.
        template_fp = pipeline_ctx.provide_or_none("brat_vis_fp")
        if template_fp is None:
            exp_root = os.path.join(input_data._get_experiment_sources_dir(),
                                    input_data.get_experiment_folder_name())
            template_fp = join(exp_root, "brat_output.tsv.gz")

        # Save results.
        with open(template_fp, "w") as output:
            output.write(template)
