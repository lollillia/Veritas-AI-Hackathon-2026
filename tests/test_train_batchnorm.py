import pytest

torch = pytest.importorskip("torch")

from train.batchnorm import use_standard_batchnorm_momentum


def test_hf_mobilenet_style_momentum_is_reset_to_pytorch_default():
    # The HF MobileNetV2 builds BatchNorm with momentum=0.997, which makes the running
    # statistics equal to the last batch's, so a saved model evaluates erratically.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4, momentum=0.997),
        torch.nn.Sequential(torch.nn.BatchNorm2d(4, momentum=0.997)),
    )
    changed = use_standard_batchnorm_momentum(model)
    assert changed == 2
    assert [m.momentum for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)] == [0.1, 0.1]


def test_running_stats_average_over_many_batches_after_the_fix():
    bn = torch.nn.BatchNorm2d(1, momentum=0.997)
    model = torch.nn.Sequential(bn)
    use_standard_batchnorm_momentum(model)
    model.train()
    for value in [0.0] * 50 + [10.0]:          # one outlier batch at the end
        model(torch.full((8, 1, 4, 4), value) + torch.randn(8, 1, 4, 4) * 0.01)
    assert bn.running_mean.item() < 5.0        # not dominated by the last batch
