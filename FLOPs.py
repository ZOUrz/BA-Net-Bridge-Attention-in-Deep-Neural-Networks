import models
import torch
from torchsummary import summary
from thop import profile
from torchstat import stat
from fvcore.nn import FlopCountAnalysis

model_name = 'swin_small_patch4_window7_224'
model = models.__dict__[model_name](num_classes=1000)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The parameters of the model:{:,}'.format(n_parameters))

inputs = torch.rand(1, 3, 224, 224)
flops, params = profile(model, inputs=(inputs, ))
print(f"{model_name}: FLOPs: {flops/1e9:.2f}G, Param: {n_parameters/1e6:.2f}M")
# x = torch.ones(10, 3, 224, 224)
# flops = FlopCountAnalysis(model, x)
# print(flops.total())
# stat(model, (3, 224, 224))


