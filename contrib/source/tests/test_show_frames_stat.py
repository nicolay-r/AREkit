import sys

sys.path.append('../../../../')

from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentiframes.stat import about_version


about_version(version=RuSentiFramesVersions.V20)
