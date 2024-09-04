import argparse
import os
import sys

from PIL import Image, ImageOps
import cv2
import numpy as np
import torch


sys.path.append(".")
from networks.CNN_Net import CustomNet
import load_model

from libkaraaba import fileio, common_utils as cu, cv_utils

def pil_grayscale(image_rgb_obj):
    image_gs = ImageOps.grayscale(image_rgb_obj)
    rgbimg = Image.merge("RGB", (image_gs, image_gs, image_gs))
    return rgbimg

def yolo_crop_file_standalone(img_path, min_wh2image_ratio=0.2):
    original_dir = os.getcwd()
    lib_path = "/home/karaaba/workspace/YOLOv4-Pytorch/"
    os.chdir(lib_path)
    sys.path.insert(0, lib_path)

    cfg = "./configs/yolov3.cfg"
    class_names = "./configs/coco.names"
    weights = "./weights/yolov3.weights"
    # from utilities.constants import *
    from utilities.constants import NO_BENCHMARK, SEPARATOR

    # from utilities.arguments import parse_detect_args
    from utilities.configs import parse_config, parse_names
    from utilities.weights import load_weights

    from utilities.devices import gpu_device_name, get_device, use_cuda
    from utilities.images import load_image, get_bboxes_xywh_with_class_filters
    from utilities.inferencing import inference_on_image, inference_video_to_video

    """
    ----------
    Author: Damon Gwinn (gwinndr)

    """

    # no_grad disables autograd so our model runs faster
    with torch.no_grad():
        print("Parsing config into model...")
        model = parse_config(cfg)
        if (model is None):
            return

        model = model.to(get_device())
        model.eval()

        # Network input dim
        if (model.net_block.width != model.net_block.height):
            print("Error: Width and height must match in [net]")
            return

        network_dim = model.net_block.width

        # Letterboxing
        letterbox = True

        print("Parsing class names...")
        class_names = parse_names(class_names)
        if (class_names is None):
            return

        print("Loading weights...")
        load_weights(model, weights)

        image = load_image(img_path)
        im_h,im_w = image.shape[:2]
        if (image is None):
            return

        detections = inference_on_image(model, image, network_dim, obj_thresh=0.9, letterbox=letterbox)

        print("detections,", len(detections))
        bboxess_probs = get_bboxes_xywh_with_class_filters(detections, class_names,
                                                           class_filters=["car"])
        # output_image = draw_detections(detections, image, class_names, verbose_output=True,
        #                                class_filters="car")

        # cv2.imwrite(args.output, output_image)

        # cv2.imshow("Detections", output_image)
        # cv2.waitKey(0)

    os.chdir(original_dir)

    min_w = min_wh2image_ratio * im_w #minimum w and h of the bbox
    min_h = min_wh2image_ratio * im_h  # minimum w and h of the bbox
    print("min_w,min_h", min_w, min_h)
    bboxess_probs = [(bbox,probability) for bbox,probability in bboxess_probs if probability>0.9]
    bboxess_probs = [((x,y,w,h), probability) for (x,y,w,h), probability in bboxess_probs if w>=min_w and h>min_h]

    return bboxess_probs

# def load_model(*, model_path, model_params):
#     properties = {}
#     for parameter in model_params:
#         if parameter.startswith("classes"):
#             properties["classes"] = parameter.split(':')[1].strip().split(',')
#         elif parameter.startswith("number of classes"):
#             properties["number of classes"] = int(parameter.split(':')[1].strip())
#         elif parameter.startswith("adaptive pool output"):
#             properties["adaptive pool output"] = tuple(map(int, parameter.split(':')[1].strip().split(",")))
#         elif parameter.startswith("resolution"):
#             properties["resolution"] = parameter.split(':')[1].strip()
#
#     adp_pool = properties["adaptive pool output"]
#     num_classes = properties["number of classes"]
#     classes = properties["classes"]
#     w, h = list(map(int, properties["resolution"].split("x")))
#
#     try:
#         model_name = properties["model name"]
#         print("model name:", properties["model name"])
#     except:
#         print("model name:", "resnet18")
#         model_name = "resnet18"
#
#     mymodel = CustomNet(num_classes=num_classes, adaptive_pool_output=adp_pool, model_name=model_name)
#     # mymodel.load_state_dict(torch.load(model_path))
#
#     from torch import nn, optim
#     try:
#         checkpoint = torch.load(model_path)
#         mymodel.load_state_dict(checkpoint['model_state_dict'])
#
#         optimizer = optim.Adam(mymodel.parameters(), lr=0.0)
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#         print("checkpoint")
#     except:
#         mymodel.load_state_dict(torch.load(model_path))
#         print("exception: classic")
#
#     mymodel.eval()
#     torch.no_grad()
#
#     return mymodel,(w, h),classes

def put_text_top(image, text, point, color=(100, 0, 100), scale=0.5):
    # h,w = image.shape[:2]
    cv2.putText(image, org=point,
           text=text, fontScale=scale,
           fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color)

def show_prediction_result_2models(*, file_path, img_obj, classes, model1, model2, resolution1, resolution2, bboxes_probs, resize_w=300):
    w1, h1 = resolution1
    w2, h2 = resolution2

    img = np.array(img_obj)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(bboxes_probs) > 0:
       (xx, yy, ww, hh), conf_detection = bboxes_probs[0]
       x1, y1, x2, y2 = cv_utils.convert_bbox_xywh2xyxy((xx, yy, ww, hh))
       cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)

       img_obj_crop = img_obj.crop((x1, y1, x2, y2))
       return_info1 = predict_image(img_obj_crop, file_path, model1, res=(w1, h1),
                               labels=classes, min_prob_threshold=0.75, target_label="", print_info=True)
       predicted_numpy1, label1, conf_class1, score1 = return_info1

       return_info2 = predict_image(img_obj_crop, file_path, model2, res=(w2, h2),
                                    labels=classes, min_prob_threshold=0.75, target_label="", print_info=True)
       predicted_numpy2, label2, conf_class2, score2 = return_info2
       img = cv_utils.resize_width(img, resize_w)
       new_img = pad_image(img, W_offset_right=80)
       # H,W = img.shape[:2]
       font_scale = 0.4
       line_size = 20
       put_text_top(new_img, label1, point=(img.shape[1],line_size), scale=font_scale)
       put_text_top(new_img, "p:" + str(round(conf_class1, 2)), point=(img.shape[1],line_size*3), scale=font_scale)

       put_text_top(new_img, label2, point=(img.shape[1], line_size*5), scale=font_scale)
       put_text_top(new_img, "p:" + str(round(conf_class2, 2)), point=(img.shape[1], line_size*7), scale=font_scale)

       cv2.imshow(os.path.basename(file_path), new_img)

def pad_image(img_cv, W_offset_right=65):
    H,W = img_cv.shape[:2]
    new_shape = (H, W+W_offset_right, 3)

    new_img = np.zeros(shape=new_shape).astype(np.uint8)
    new_img[0:img_cv.shape[0], 0:img_cv.shape[1]] = img_cv

    return new_img



def show_prediction_result(*, file_path, img_obj, classes, model, model_name, resolution, bboxes_probs, resize_w=300):
    w,h = resolution

    img = np.array(img_obj)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(bboxes_probs) > 0:
       (xx, yy, ww, hh), conf_detect = bboxes_probs[0]
       x1, y1, x2, y2 = cv_utils.convert_bbox_xywh2xyxy((xx, yy, ww, hh))

       img_obj_crop = img_obj.crop((x1, y1, x2, y2))
       return_val = predict_image(img_obj_crop, file_path, model, res=(w, h),
                               labels=classes, min_prob_threshold=0.75, target_label="", print_info=True)

       predicted_numpy, label, conf_class, score = return_val
       cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)

       img = cv_utils.resize_width(img, resize_w)
       H,W = img.shape[:2]
       new_shape = (H, W+65, 3)
       # diff = H - W
       new_img = np.zeros(shape=new_shape).astype(np.uint8)
       new_img[0:img.shape[0], 0:img.shape[1]] = img
       print("W",W)
       print("new img W", new_shape)
       font_scale=0.3
       put_text_top(new_img, label, point=(W,20), scale=font_scale)
       put_text_top(new_img, "p:" + str(round(conf_class, 2)), point=(W,40), scale=font_scale)
       cv2.imshow(label + " p" + str(round(conf_class, 2))+" m "+model_name, new_img)
    else:
       return_val = predict_image(img_obj, file_path, model, res=(w, h),
                                   labels=classes, min_prob_threshold=0.75, target_label="", print_info=True)

       predicted_numpy, label, conf_class, score = return_val
       cv2.imshow(label+" p" + str(round(conf_class, 2))+" m "+model_name, cv_utils.resize_width(img, resize_w))

def main_detect_and_classify_2models():
    parser = argparse.ArgumentParser(description='Test ResNet-18 on custom dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--model-path1', type=str, required=True, help='path of the model')
    parser.add_argument('--model-path2', type=str, required=True, help='path of the model')
    parser.add_argument('--preprocess', type=str, default="None", help='preprocess options: Example Flip,GS')

    args = parser.parse_args()

    model_params1 = open(args.model_path1[:-4] + ".txt").readlines()
    model_params2 = open(args.model_path2[:-4] + ".txt").readlines()
    # bboxes_probs = yolo_crop_file_standalone(img_path=args.image, min_wh2image_ratio=0.2)

    model1, (w1, h1), classes1 = load_model.load_model_weights(model_path=args.model_path1, model_params=model_params1)
    model2, (w2, h2), classes2 = load_model.load_model_weights(model_path=args.model_path2, model_params=model_params2)

    # img_obj = Image.open(args.image)
    print("w h", w1,h1)
    print("w2 h2", w2, h2)

    bboxes_probs = yolo_crop_file_standalone(img_path=args.image, min_wh2image_ratio=0.2)

    img_obj = Image.open(args.image)

    # print("preprocess",preprocess)
    if not args.preprocess is None:
        if "Flip".lower() in args.preprocess.lower():
            img_obj = img_obj.transpose(Image.FLIP_LEFT_RIGHT)

        if "GS".lower() in args.preprocess.lower():
            img_obj = pil_grayscale(img_obj)

    show_prediction_result_2models(file_path=args.image, img_obj=img_obj, classes=classes1, resolution1=(w1,h1),resolution2=(w2,h2),
                                   model1=model1,model2=model2,
                            bboxes_probs=bboxes_probs, resize_w=660)

    # show_prediction_result(file_path=args.image, img_obj=img_obj, classes=classes1, resolution=(w,h), model=model1,model_name="1",
    #                        bboxes_probs=bboxes_probs, resize_w=660)
    #
    # show_prediction_result(file_path=args.image, img_obj=img_obj, classes=classes2, resolution=(w2, h2), model=model2, model_name="2",
    #                        bboxes_probs=bboxes_probs, resize_w=660)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main_detect_and_classify():
    parser = argparse.ArgumentParser(description='Test ResNet-18 on custom dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--model-path', type=str, required=True, help='path of the model')
    parser.add_argument('-pred-type', '--prediction-type', default="classification", type=str, help='path to the model')
    # parser.add_argument('--the-class', dest="label", default="", type=str, help='class name if there is only one class')
    parser.add_argument('--preprocess', type=str, default="None", help='preprocess options: Example Flip,GS')

    args = parser.parse_args()

    model_params = open(args.model_path[:-4] + ".txt").readlines()

    bboxes_probs = yolo_crop_file_standalone(img_path=args.image, min_wh2image_ratio=0.2)


    mymodel,(w,h),classes = load_model.load_model(model_path=args.model_path, model_params=model_params)

    img_obj = Image.open(args.image)

    show_prediction_result(file_path=args.image, img_obj=img_obj, classes=classes, resolution=(w,h), model_name="", model=mymodel,
                           preprocess=args.preprocess,
                           bboxes_probs=bboxes_probs, resize_w=600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main_detect_and_classify_():
    parser = argparse.ArgumentParser(description='Test ResNet-18 on custom dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--model-path', type=str, required=True, help='path of the model')
    parser.add_argument('-pred-type', '--prediction-type', default="classification", type=str, help='path to the model')
    # parser.add_argument('--the-class', dest="label", default="", type=str, help='class name if there is only one class')
    parser.add_argument('--preprocess', type=str, default="None", help='preprocess options: Example Flip,GS')

    args = parser.parse_args()

    model_params = open(args.model_path[:-4] + ".txt").readlines()



    bboxes_probs = yolo_crop_file_standalone(img_path=args.image, min_wh2image_ratio=0.2)

    # adp_pool = properties["adaptive pool output"]
    # num_classes = properties["number of classes"]
    # classes = properties["classes"]
    # w, h = list(map(int, properties["resolution"].split("x")))

    # mymodel = CustomNet(num_classes=num_classes, adaptive_pool_output=adp_pool)
    # mymodel.load_state_dict(torch.load(args.model_path))
    # mymodel.eval()
    # torch.no_grad()

    mymodel,(w,h),classes = load_model(model_path=args.model_path, model_params=model_params)

    img_obj = Image.open(args.image)

    show_prediction_result(file_path=args.image, img_obj=img_obj, classes=classes, resolution=(w,h), model_name="", model=mymodel,
                           preprocess=args.preprocess,
                           bboxes_probs=bboxes_probs, resize_w=600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def classify_image_wrap()

def predict_image(image_obj, img_filename, model, labels=("acura", "alpha romeo"),
                  target_label="acura", res=(128, 128),
                  min_prob_threshold=0.75, print_info=False):
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F
    # print("labels",labels)
    import torchvision
    from scipy.special import softmax

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("device is", device)
    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)
    image = image_obj

    if image.mode == 'RGBA':
       image = image.convert('RGB')

    model.to(device=device)
    image = image.resize(res)
    data = TF.to_tensor(image)
    data = normalizer(data)
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)
    # print("output", output.cpu().detach().numpy()[0])
    _, predicted = torch.max(output.data, 1)
    predicted_numpy = predicted.cpu().detach().numpy()

    # clamped_outputs = output.clamp(0, 1)

    raw_output = output.cpu().detach().numpy()

    probabilities = np.round(softmax(raw_output),2)
    confidence_value = np.max(probabilities)

    final_predicted_value = labels[predicted]
    score = None

    if min_prob_threshold > confidence_value:
       final_predicted_value = "unsure"
       if print_info:
          print("unsure", end=",")
          print("prob", probabilities, "conf:", confidence_value, "pred:", predicted_numpy,"real pred val",
             final_predicted_value, "Target:", target_label,  os.path.basename(img_filename))
    else:
        # print(os.path.basename(filepath), end=" ")
        if print_info:
           print ("prob",probabilities,"confidence:",confidence_value, "pred:",predicted_numpy,"real pred val",
               final_predicted_value, "Target:", target_label,  os.path.basename(img_filename))
        score = True if final_predicted_value==target_label else False


    return predicted_numpy,labels[predicted],confidence_value,score   #this part is used for the single main

import sys

if __name__ == '__main__':
    commands = ['single','multi']

    if len(sys.argv)==1:
       print ('options are',commands)
       exit()

    inputCommand = sys.argv[1]
    commandArgs = sys.argv[1:]

    print("inputCommand, commandArgs:", inputCommand, commandArgs)

    if sys.argv[1] == commands[0]:
       sys.argv = commandArgs
       main_detect_and_classify()
    elif sys.argv[1]==commands[1]:
       sys.argv = commandArgs
       main_detect_and_classify_2models()
