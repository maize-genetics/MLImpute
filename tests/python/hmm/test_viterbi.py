import unittest
import torch
import torch.nn.functional as F
from python.hmm.viterbi import build_pair_states, make_diploid_emissions, make_diploid_transitions, viterbi_decode
import math


class TestHMM(unittest.TestCase):
    def setUp(self):
        self.N = 3
        self.seq_length = 5
        self.pair_states = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.log_em = torch.tensor([[0.1, 0.2, 0.3],
                                    [0.1, 0.2, 0.3],
                                    [0.1, 0.2, 0.3],
                                    [0.1, 0.2, 0.3],
                                    [0.1, 0.2, 0.3]])

        self.log_A = torch.tensor([[0.5, 0.25, 0.25],
                                   [0.25, 0.5, 0.25],
                                   [0.25, 0.25, 0.5]])

        self.log_start = torch.tensor([0.5, 0.25, 0.25])

        self.dip_log_em = torch.tensor([[0.2, 0.3, 0.4, 0.4, 0.5, 0.6],
                                        [0.2, 0.3, 0.4, 0.4, 0.5, 0.6],
                                        [0.2, 0.3, 0.4, 0.4, 0.5, 0.6],
                                        [0.2, 0.3, 0.4, 0.4, 0.5, 0.6],
                                        [0.2, 0.3, 0.4, 0.4, 0.5, 0.6]])

        self.dip_log_A = torch.tensor([[1.0, 0.75, 0.75, 0.5, 0.5, 0.5],
                                       [0.75, 1.0, 0.75, 0.75, 0.5, 0.5],
                                       [0.75, 0.75, 1.0, 0.5, 0.75, 0.75],
                                       [0.5, 0.75, 0.5, 1.0, 0.75, 0.5],
                                       [0.5, 0.5, 0.75, 0.75, 1.0, 0.75],
                                       [0.5, 0.5, 0.75, 0.5, 0.75, 1.0]
                                       ])

        self.dip_log_start = torch.tensor([math.log(1/6), math.log(1/6), math.log(1/6), math.log(1/6), math.log(1/6), math.log(1/6)])


    def test_build_pair_states(self):
        pairs = build_pair_states(n_parents=self.N)
        self.assertEqual(len(pairs), len(self.pair_states))
        self.assertEqual(len(pairs[0]), 2)
        self.assertEqual(pairs, self.pair_states)

    def test_make_diploid_emissions(self):
        dip_em = make_diploid_emissions(self.log_em, self.pair_states)
        self.assertEqual(dip_em.shape, (self.seq_length, len(self.pair_states)))
        self.assertTrue(torch.equal(dip_em, self.dip_log_em))

    def test_make_diploid_transitions(self):
        dip_tran = make_diploid_transitions(self.log_A, self.pair_states)
        self.assertEqual(dip_tran.shape, (len(self.pair_states), len(self.pair_states)))
        self.assertTrue(torch.equal(dip_tran, self.dip_log_A))

    def test_viterbi_decode(self):
        hap_path = viterbi_decode(self.log_em, self.log_A, self.log_start)
        print(hap_path)
        self.assertEqual(len(hap_path), self.seq_length)
        self.assertFalse(5 in hap_path)
        self.assertFalse(4 in hap_path)
        self.assertFalse(3 in hap_path)

        dip_path = viterbi_decode(self.dip_log_em, self.dip_log_A, self.dip_log_start)
        print(dip_path)
        self.assertEqual(len(dip_path), self.seq_length)



if __name__ == "__main__":
    unittest.main()