import unittest
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector


class TestsQuorum(unittest.TestCase):
    def test_t0s_matching(self):
        quorum_selector = QuorumSnrBorderCorrectedSignalSelector()
        assert quorum_selector.matches_t0(1, 1, 1, 1)
        assert quorum_selector.matches_t0(1, 2, 1, -2)
        assert quorum_selector.matches_t0(1, 3, 1, -3)
        assert quorum_selector.matches_t0(3, 2.66, 1, 3)
        assert quorum_selector.matches_t0(2, 2.5, 1, 2)
        assert quorum_selector.matches_t0(1, 1.01, 1, 1)
        assert quorum_selector.matches_t0(1, 2.01, 1, -2)
        assert quorum_selector.matches_t0(1, 3.01, 1, -3)
        assert quorum_selector.matches_t0(2, 1.505, 1, 2)
        assert quorum_selector.matches_t0(0.99, 1, 1, 1)
        assert quorum_selector.matches_t0(0.99, 2, 1, -2)
        assert quorum_selector.matches_t0(0.99, 3, 1, -3)
        assert quorum_selector.matches_t0(2.99, 2.66, 1, 3)
        assert quorum_selector.matches_t0(2.0, 2.51, 1, 2)
        assert not quorum_selector.matches_t0(2.0, 2.6, 1, 2)
        assert not quorum_selector.matches_t0(1000.0, 1005.0, 3, -2)

    def test_periods_matching(self):
        quorum_selector = QuorumSnrBorderCorrectedSignalSelector()
        assert quorum_selector.multiple_of(1, 3)
        assert quorum_selector.multiple_of(1, 2)
        assert quorum_selector.multiple_of(1, 1)
        assert quorum_selector.multiple_of(3, 1)
        assert quorum_selector.multiple_of(2, 1)
        assert quorum_selector.multiple_of(1, 3.01)
        assert quorum_selector.multiple_of(1, 2.01)
        assert quorum_selector.multiple_of(1.01, 1)
        assert quorum_selector.multiple_of(2.01, 1)
        assert quorum_selector.multiple_of(1.01, 1.02)
        assert not quorum_selector.multiple_of(3, 2)
        assert not quorum_selector.multiple_of(2, 3)
        assert not quorum_selector.multiple_of(1.99, 4.06)

    def test_harmonic(self):
        quorum_selector = QuorumSnrBorderCorrectedSignalSelector()
        assert quorum_selector.is_harmonic(1000.0, 1002.0, 4, 2)
        assert quorum_selector.is_harmonic(1000.0, 1004.0, 6, 2)
        assert quorum_selector.is_harmonic(1002.0, 1004.0, 2, 4)
        assert quorum_selector.is_harmonic(1004.0, 1000.0, 2, 6)
        assert quorum_selector.is_harmonic(1000.0, 999.99, 2, 2)
        assert not quorum_selector.is_harmonic(1000.0, 996.99, 2, 2)
        assert not quorum_selector.is_harmonic(1000.0, 996.99, 2, 4)


if __name__ == '__main__':
    unittest.main()
