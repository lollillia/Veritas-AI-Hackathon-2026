def use_standard_batchnorm_momentum(model, momentum: float = 0.1) -> int:
    """Reset every BatchNorm layer's momentum to the PyTorch default.

    Hugging Face's MobileNetV2 creates its BatchNorm layers with momentum=0.997, which in
    PyTorch makes the running statistics (almost) equal to the last batch's. After training
    on mixed-source data the saved model then evaluates erratically (we measured 55% vs 94%
    accuracy for the same weights). Returns how many layers were changed.
    """
    import torch

    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.momentum != momentum:
            module.momentum = momentum
            changed += 1
    return changed
