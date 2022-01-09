from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions, RuSentiFramesVersionsService
from examples.network.args.base import BaseArg


class RuSentiFramesVersionArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return RuSentiFramesVersionsService.get_type_by_name(args.frames_version)

    @staticmethod
    def add_argument(parser):

        default_name = RuSentiFramesVersionsService.get_name_by_type(
            RuSentiFramesVersions.V20)

        parser.add_argument('--frames-version',
                            dest='frames_version',
                            type=str,
                            default=default_name,
                            choices=list(RuSentiFramesVersionsService.iter_supported_names()),
                            nargs='?',
                            help='Version of RuSentiFrames collection (Default: {})'.format(default_name))
