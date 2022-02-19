from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parsed import BaseParsedText


class BaseTextParser(BasePipeline):

    def run(self, input_data, params_dict=None):
        output_data = super(BaseTextParser, self).run(input_data=input_data,
                                                      params_dict=params_dict)

        return BaseParsedText(terms=output_data)
