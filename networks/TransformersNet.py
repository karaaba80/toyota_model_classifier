
import torch.nn as nn
import timm

class TransformerVIT(nn.Module):
    def __init__(self, num_classes):
        super(TransformerVIT, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
