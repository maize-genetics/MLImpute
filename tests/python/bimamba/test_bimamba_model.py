import unittest
import torch
import torch.nn.functional as F
from python.bimamba.bimamba_model import SNPLoss, SNPLossSmooth, SNPLossSmoothAll, BiMambaSmooth


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

    def test_zero(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertLess(loss.item(), 1e-6)

    def test_nonzero(self):
        # make logits oppositely match targets: large - for 1s, large + for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(-20.), torch.tensor(20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertGreater(loss.item(), 1e-6)


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

    def test_zero(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [1., 0., 1.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertLess(loss.item(), 1e-6)

    def test_nonzero_smooth(self):
        # make logits oppositely match targets: large - for 1s, large + for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [1., 0., 1.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(-20.), torch.tensor(20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertGreater(loss.item(), 1e-6)

    def test_nonzero_nonsmooth(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertGreater(loss.item(), 1e-6)


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

    def test_zero(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [1., 0., 1.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
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
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertGreater(loss.item(), 1e-6)

    def test_nonzero_nonsmooth(self):
        # make logits perfectly match targets: large + for 1s, large - for 0s
        targets = torch.tensor([[
            [1., 0., 1.],
            [0., 1., 0.],
        ]])
        logits = torch.where(targets == 1., torch.tensor(20.), torch.tensor(-20.))
        mask = torch.tensor([[True, True]])
        loss = self.loss_fn(logits, targets, mask)
        self.assertGreater(loss.item(), 1e-6)


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

    def test_forward_shape_hidden(self):
        self.model.to(self.device)
        self.model.eval()
        input = torch.randn(self.batch_size, self.window_size, self.num_classes, device=self.device)
        hidden_states = self.model(input, hidden=True)
        self.assertEqual(hidden_states.shape, (torch.Size([self.batch_size, self.window_size, self.d_model])))

    def test_load_pretrained(self):
        before = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        checkpoint = "src/bimamba_model.pth"
        self.model.load_state_dict(torch.load(checkpoint))
        after = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        # verify something actually changed
        changed = any(not torch.allclose(before[k], after[k]) for k in before.keys() if k in after)
        self.assertTrue(changed, "Weights did not change after loading checkpoint")


if __name__ == "__main__":
    unittest.main()