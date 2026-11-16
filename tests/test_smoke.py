import re

import grid


def test_package_exposes_a_version():
    assert grid.__version__


def test_version_looks_like_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", grid.__version__)
