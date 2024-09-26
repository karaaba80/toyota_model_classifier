import sys
import torch

from networks.TransformersNet import TransformerVIT

sys.path.append(".")
from networks.CNN_Net import ResNet

def load_model_parameters(model_params):
    """
    ----------
    Author: M. Faik Karaaba (karaaba80)
    ----------
    loading model meta parameters like classes, image resolution, model name
    ----------
    """

    properties = {}
    for parameter in model_params:
        if parameter.startswith("classes"):
            properties["classes"] = parameter.split(':')[1].strip().split(',')
        elif parameter.startswith("number of classes"):
            properties["number of classes"] = int(parameter.split(':')[1].strip())
        elif parameter.startswith("adaptive pool output"):
            properties["adaptive pool output"] = tuple(map(int, parameter.split(':')[1].strip().split(",")))
        elif parameter.startswith("resolution"):
            properties["resolution"] = parameter.split(':')[1].strip()
        elif parameter.startswith("model name"):
            properties["model name"] = parameter.split(':')[1].strip()

    return properties

def load_model_weights(*, model_path, model_params, device):
    """
    ----------
    Author: M. Faik Karaaba (karaaba80)
    ----------
    loading model weights, depending model params, it loads different architechtures like resnet or transformer VIT
    ----------
    """

    properties = load_model_parameters(model_params)
    try:
        model_name = properties["model name"]
        print("model name:", properties["model name"])
    except:
        print("model name:", "resnet18")
        model_name = "resnet18"

    num_classes = properties["number of classes"]
    classes = properties["classes"]
    w, h = list(map(int, properties["resolution"].split("x")))

    if model_name.startswith("resnet"):
       adp_pool = properties["adaptive pool output"]
       print("resolution", str(w)+"x"+str(h))
       mymodel = ResNet(num_classes=num_classes, adaptive_pool_output=adp_pool, model_name=model_name)

    else:
       num_classes = properties["number of classes"]
       mymodel = TransformerVIT(num_classes=num_classes)

    mymodel.load_state_dict(torch.load(model_path,map_location=device))

    mymodel.eval()
    torch.no_grad()

    return mymodel,(w, h),classes


