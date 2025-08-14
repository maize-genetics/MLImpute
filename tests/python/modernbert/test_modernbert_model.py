import unittest
import torch

from src.python.modernBERT.modernBERT_model import SNPLoss, SNPLossSmoothAll, BERTImpute, BERTImputeConfig


class TestSNPLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = SNPLoss()
        self.logits = torch.randn(4, 10, 8)
        self.unmasked_input = torch.randint(0, 2, (4, 10, 8)).float()
        self.loss = self.loss_fn(self.logits, self.unmasked_input)

    def test_shapes_match(self):
        self.assertEqual(self.loss.shape, torch.Size([]))

    def test_not_nan(self):
        self.assertFalse(torch.isnan(self.loss))


class TestSNPLossSmoothAll(unittest.TestCase):
    def setUp(self):
        self.loss_fn = SNPLossSmoothAll()
        self.logits = torch.randn(4, 10, 8)
        self.unmasked_input = torch.randint(0, 2, (4, 10, 8)).float()
        self.loss = self.loss_fn(self.logits, self.unmasked_input)

    def test_shapes_match(self):
        self.assertEqual(self.loss.shape, torch.Size([]))

    def test_not_nan(self):
        self.assertFalse(torch.isnan(self.loss))

class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.window_size = 512
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = 8e-4
        self.learning_rate_decay = "none"
        self.torch_compile = "no"
        self.batch_size = 64
        self.num_classes = 25

        config = BERTImputeConfig(
            architecture="encoder-only",
            max_sequence_length=self.window_size,
        )
        self.model = BERTImpute(
            config,
            learning_rate=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            torch_compile=self.torch_compile == "yes",
        )

    def test_forward_shape(self):
        self.model.to(self.device)
        self.model.eval()
        input = torch.randn(self.batch_size, self.window_size, self.num_classes, device=self.device)
        outputs = self.model(input)
        self.assertEqual(outputs.shape, (torch.Size([self.batch_size, self.window_size, self.num_classes])))

if __name__ == "__main__":
    unittest.main()