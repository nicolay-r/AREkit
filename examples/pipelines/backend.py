from examples.brat_backend import BratBackend


# TODO. #285 reorganize in a form of a pipeline item.
def pipeline_brat_backend(output_data_filepath, sample_data_filepath,
                          obj_color_types, rel_color_types, label_to_rel, target):
    assert(isinstance(target, str))

    brat_be = BratBackend()

    template = brat_be.to_html(result_data_filepath=output_data_filepath,
                               samples_data_filepath=sample_data_filepath,
                               obj_color_types=obj_color_types,
                               rel_color_types=rel_color_types,
                               label_to_rel=label_to_rel,
                               brat_url="http://localhost:8001/")

    # Save results.
    with open(target, "w") as output:
        output.write(template)
