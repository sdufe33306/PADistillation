# utils/hooks.py

import torch

def get_feature_maps(model, layer_names):
    """
    获取模型中指定层的特征图。

    Args:
        model (torch.nn.Module): 模型。
        layer_names (list): 需要提取特征图的层名称。

    Returns:
        dict: 特征图字典，键为层名称，值为特征图。
        list: 注册的钩子列表。
    """
    feature_maps = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    return feature_maps, hooks
