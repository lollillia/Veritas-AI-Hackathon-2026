import pytest

torch = pytest.importorskip("torch")

from train.fusion import FusionClassifier


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        self.bn = torch.nn.BatchNorm2d(8, momentum=0.997)

    def forward(self, pixel_values):
        x = self.bn(self.conv(pixel_values)).mean(dim=(2, 3))
        return type("Out", (), {"pooler_output": x})()


def _batch(n=4):
    return torch.randn(n, 3, 16, 16), torch.randn(n, 25), torch.tensor([0, 1, 2, 3][:n])


def test_logits_shape_and_loss_with_weather():
    m = FusionClassifier(FakeBackbone(), feat_dim=8, num_labels=4, n_tab=25, use_tab=True)
    px, tab, y = _batch()
    out = m(pixel_values=px, tab=tab, labels=y)
    assert out["logits"].shape == (4, 4) and out["loss"].ndim == 0


def test_image_only_model_ignores_weather():
    torch.manual_seed(0)
    m = FusionClassifier(FakeBackbone(), feat_dim=8, num_labels=4, n_tab=25, use_tab=False).eval()
    px, tab, _ = _batch()
    a = m(pixel_values=px, tab=tab)["logits"]
    b = m(pixel_values=px, tab=torch.randn_like(tab))["logits"]
    assert torch.allclose(a, b)


def test_weather_changes_the_prediction_when_used():
    torch.manual_seed(0)
    m = FusionClassifier(FakeBackbone(), feat_dim=8, num_labels=4, n_tab=25, use_tab=True).eval()
    px, tab, _ = _batch()
    assert not torch.allclose(m(pixel_values=px, tab=tab)["logits"], m(pixel_values=px, tab=tab + 5)["logits"])
