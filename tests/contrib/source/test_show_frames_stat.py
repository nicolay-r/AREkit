import sys
import unittest

sys.path.append('../../../../')

from arekit.contrib.source.run_rusentiframes_stat import about_version
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions


class TestFramesStat(unittest.TestCase):

    def test_stat(self):
        about_version(version=RuSentiFramesVersions.V20)


if __name__ == '__main__':
    unittest.main()
