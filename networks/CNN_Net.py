import torch.nn as nn
import torchvision.models as models
import torchvision

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ResNet(nn.Module):
    def __init__(self, num_classes, model_name="resnet18", adaptive_pool_output=(1, 1), pretrained=True):  # resnet50

        super(ResNet, self).__init__()

        in_features_size = 256
        try:
            if model_name.lower()=="resnet18":
               in_features_size = 256
               self.custom_model = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-3])
               print("model is resnet18")
            elif model_name.lower()=="resnet50":
               in_features_size = 1024
               self.custom_model = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-3])
               print("model is resnet50")
            #self.custom_model = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-3])


            self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=adaptive_pool_output)

            new_size = in_features_size * adaptive_pool_output[0] * adaptive_pool_output[1]
            self.fc = nn.Linear(new_size, num_classes)
            self.adaptive_pool_output = adaptive_pool_output
            self.num_classes = num_classes
        except Exception as E:
           print("exception", E)


    def forward(self, x):
        try:
            x = self.custom_model(x)
            x = self.adaptiveAvgPool2d(x)

            x = x.view(x.size(0), -1)
            x = self.fc(x)
        except Exception as E:
            print("Exception",E)
            print(self.custom_model)
            exit()
        return x