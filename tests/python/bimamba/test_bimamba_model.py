import unittest
import torch
from src.python.bimamba.bimamba_model import SNPLoss, SNPLossSmooth, SNPLossSmoothAll

class TestSNPLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = SNPLoss()
        self.logits = torch.randn(4, 10, 8)
        self.unmasked_input = torch.randint(0, 2, (4, 10, 8)).float()
        self.masked = torch.randint(0, 2, (4, 10), dtype=torch.bool)
        self.loss = self.loss_fn(self.logits, self.unmasked_input, self.masked)

    def test_shapes_match(self):
        self.assertEqual(self.loss.shape, torch.Size([]))

    def test_not_nan(self):
        self.assertFalse(torch.isnan(self.loss))


class TestSNPLossSmooth(unittest.TestCase):
    def setUp(self):
        self.loss_fn = SNPLossSmooth()
        self.logits = torch.randn(4, 10, 8)
        self.unmasked_input = torch.randint(0, 2, (4, 10, 8)).float()
        self.masked = torch.randint(0, 2, (4, 10), dtype=torch.bool)
        self.loss = self.loss_fn(self.logits, self.unmasked_input, self.masked)

    def test_shapes_match(self):
        self.assertEqual(self.loss.shape, torch.Size([]))

    def test_not_nan(self):
        self.assertFalse(torch.isnan(self.loss))


class TestSNPLossSmoothAll(unittest.TestCase):
    def setUp(self):
        self.loss_fn = SNPLossSmoothAll()
        self.logits = torch.randn(4, 10, 8)
        self.unmasked_input = torch.randint(0, 2, (4, 10, 8)).float()
        self.masked = torch.randint(0, 2, (4, 10), dtype=torch.bool)
        self.loss = self.loss_fn(self.logits, self.unmasked_input, self.masked)

    def test_shapes_match(self):
        self.assertEqual(self.loss.shape, torch.Size([]))

    def test_not_nan(self):
        self.assertFalse(torch.isnan(self.loss))

if __name__ == "__main__":
    unittest.main()