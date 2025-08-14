import unittest
import torch
import torch.nn.functional as F
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

    def test_zero(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        loss = self.loss_fn(logits, targets)
        self.assertLess(loss.item(), 1e-6)

    def test_nonzero(self):
        # make logits oppositely match targets: large - for 1s, large + for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(-20.), torch.tensor(20.))
        loss = self.loss_fn(logits, targets)
        self.assertGreater(loss.item(), 1e-6)

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

    def test_zero(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [1., 0., 1.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        loss = self.loss_fn(logits, targets)
        CLAMP = 10.0
        eps = F.softplus(torch.tensor(-CLAMP)).item()
        self.assertLess(loss.item(), eps + 1e-6)

    def test_nonzero_smooth(self):
        # make logits oppositely match targets: large - for 1s, large + for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [1., 0., 1.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(-20.), torch.tensor(20.))
        loss = self.loss_fn(logits, targets)
        self.assertGreater(loss.item(), 1e-6)

    def test_nonzero_nonsmooth(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        loss = self.loss_fn(logits, targets)
        self.assertGreater(loss.item(), 1e-6)

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

    def test_load_pretrained(self):
        before = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        checkpoint = "src/modernbert.pth"
        self.model.load_state_dict(torch.load(checkpoint))
        after = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        # verify something actually changed
        changed = any(not torch.allclose(before[k], after[k]) for k in before.keys() if k in after)
        self.assertTrue(changed, "Weights did not change after loading checkpoint")

if __name__ == "__main__":
    unittest.main()