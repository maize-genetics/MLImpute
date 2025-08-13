import unittest
import torch

from src.python.bimamba.bimamba_model import SNPLoss, SNPLossSmooth, SNPLossSmoothAll, BiMambaSmooth


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

class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.window_size = 512
        self.num_classes = 25
        self.input_dim = 25
        self.d_model = 128
        self.n_layer = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BiMambaSmooth(self.input_dim, self.d_model, self.num_classes, self.n_layer, d_conv=4, device=None, dtype=None, lambda_smooth=0.2)

    def test_forward_shape(self):
        self.model.to(self.device)
        self.model.eval()
        input = torch.randn(self.batch_size, self.window_size, self.num_classes, device=self.device)
        outputs, mask = self.model(input)
        self.assertEqual(outputs.shape, (torch.Size([self.batch_size, self.window_size, self.num_classes])))

if __name__ == "__main__":
    unittest.main()