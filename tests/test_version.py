from annotuneagent._version import __version__, version_tuple


def test_version_tuple():
    assert __version__ == "0.1.1"
    assert version_tuple == (0, 1, 1)
