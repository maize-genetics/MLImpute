import unittest
import torch
import torch.nn.functional as F
from python.autoencoder.autoencoder_model import BCELoss, AutoEncoder


class TestBCELoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = BCELoss()
        self.logits = torch.randn(4, 10, 8)
        self.targets = torch.randint(0, 2, (4, 10, 8)).float()
        self.loss = self.loss_fn(self.logits, self.targets)

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


class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.parents = 25
        self.window_size = 512
        self.hidden_dim = 512
        self.bottleneck_dim = 51
        self.dropout = 0.1
        self.batch_size = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoEncoder(self.parents, self.window_size, self.hidden_dim, self.bottleneck_dim, self.dropout)

    def test_forward_shape(self):
        self.model.to(self.device)
        self.model.eval()
        input = torch.randn(self.batch_size, self.window_size, self.parents, device=self.device)
        outputs = self.model(input)
        self.assertEqual(outputs.shape, (torch.Size([self.batch_size, self.window_size, self.parents])))

    def test_load_pretrained(self):
        before = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        checkpoint = "src/autoencoder.pth"
        self.model.load_state_dict(torch.load(checkpoint))
        after = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        # verify something actually changed
        changed = any(not torch.allclose(before[k], after[k]) for k in before.keys() if k in after)
        self.assertTrue(changed, "Weights did not change after loading checkpoint")


if __name__ == "__main__":
    unittest.main()