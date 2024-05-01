from torch import nn
from typing import Tuple, Dict

def access_grad(model: nn.Module) -> Tuple[Dict, Dict]:
    dz = [x.grad for x in model.parameters()]
    dz_named  = [x for x in model.named_parameters()] 
    mapped_grad = {dz_named_item[0]: dz_item for dz_named_item, dz_item in zip(dz_named, dz)}
    mapped_params = dict(model.named_parameters())
    return mapped_grad, mapped_params