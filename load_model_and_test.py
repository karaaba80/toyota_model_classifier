import argparse
import os
import sys

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from PIL import Image, ImageOps
import cv2
import numpy as np
import torch



import pandas as pd

sys.path.append(".")
from networks.CNN_Net import CustomNet
from networks.TransformersNet import TransformerVIT
import load_model

from libkaraaba import fileio, common_utils as cu


def get_class_counts(class_names,all_labels):
    class_counts = dict(zip(np.unique(all_labels), np.bincount(all_labels)))
    uniques = np.unique(all_labels)
    class_name_dict = {}
    for u in uniques:
        class_name_dict[class_names[u]]=class_counts[u]

    return class_name_dict




def print_confusion_matrix(y_pred_labels, y_true_labels, classes):
    label_encoder = LabelEncoder()

    label_encoder.fit(classes)
    print("label_encoder classes",label_encoder.classes_)
    y_pred = label_encoder.transform(y_pred_labels)
    y_true = label_encoder.transform(y_true_labels)

    matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)

    print("matrix\n", matrix)
    print("\nPercentage Matrix:")

    for row in matrix:
       if sum(row) > 0:
          row_percentage = 100 * (row / sum(row))  # Calculate percentages
       else:
          row_percentage = 100 * (row / sum(row + 1))
       formatted_row = " ".join([f"{int(p):3d}%" for p in row_percentage])  # Format each percentage with alignment
       print(formatted_row, " real amount", sum(row))

    # Calculate the average accuracy
    # diagonal = np.array([d for d in np.diag(matrix) if d > 0])
    diagonal = np.array([d for d in np.diag(matrix)])
    class_counts = matrix.sum(axis=1)  # Sum of each row (total samples per class)

    class_counts = np.array([c for c in class_counts if c > 0])
    class_accuracies = diagonal / class_counts  # Accuracy per class
    average_accuracy = np.mean(class_accuracies) * 100  # Mean of class accuracies

    # Print the average accuracy
    print(f"\nAverage Accuracy (accounting for class imbalance): {average_accuracy:.1f}%")



def get_predictions(*, model, input_data, img_resolution, classes, info=False, conf_thrs=-1.0):
    file_paths,image_obj_list,target_labels = input_data
    w,h = img_resolution
    results = [predict_image(img_obj, img_filename=img_file_name, model=model,
                             res=(w, h), target_label=real_label,
                             labels=classes, min_prob_threshold=conf_thrs,
                             print_info=info)
               for img_file_name, img_obj, real_label in zip(file_paths, image_obj_list, target_labels)]

    confidences = [r[2] for r in results]
    judgements = [r[3] for r in results]  # scores
    labels = [r[1] for r in results]  # pred labels

    results_targets = [(label, conf, score, os.path.basename(img_file), real_label) for
                       (label, conf, score, img_file, real_label) in
                       zip(labels, confidences, judgements, file_paths, target_labels) if not score is None]
    correct_number = len(
        [(l, c, s, filename) for (l, c, s, filename, real_label) in results_targets if l == real_label])

    print("len labels", len(results_targets), "correct:", correct_number)
    acc = correct_number / len(results_targets)
    print("accuracy %" + str(acc * 100))

    pred_labels = [r[0] for r in results_targets]
    true_labels = [r[-1] for r in results_targets]  # real labels after filter of score
    print("pred_labels")
    return pred_labels,true_labels

def prepare_csv_data(csv_file_path, preprocess=""):
    csv_data = pd.read_csv(csv_file_path)
    file_paths = [path for path in csv_data["file_path"]]
    real_labels = [label for label in csv_data["label"]]
    image_obj_list = [Image.open(f) for f in file_paths]

    if "Flip".lower() in preprocess.lower():
        image_obj_list = [img_obj.transpose(Image.FLIP_LEFT_RIGHT) for img_obj in image_obj_list]
    if "GS".lower() in preprocess.lower():
        image_obj_list = [pil_grayscale(img_obj) for img_obj in image_obj_list]

    return file_paths,image_obj_list,real_labels

def main_csv_multi_model():
    parser = argparse.ArgumentParser(description='Test ResNet-18 on custom dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('test_csv', type=str, help='Path to the testing data directory')
    parser.add_argument('--model-path-list', type=str, nargs='+', required=True, help='path of the model')

    parser.add_argument('--preprocess', type=str, default="", help='preprocess options: Example Flip,GS')
    parser.add_argument('--conf-thrs', dest="conf_thrs", default=0.75, type=float,
                        help='confidence threshold, -1 means no threshold')
    parser.add_argument('--print-info', action='store_true', help='path of the model')
    args = parser.parse_args()
    print("aaaaa")
    model_paths = args.model_path_list

    if args.conf_thrs == -1:
        args.conf_thrs = None

    for model_path in model_paths:
        print("model:", os.path.basename(model_path))
        model_params = open(model_path[:-4] + ".txt").readlines()

        model, (w, h), classes = load_model.load_model_cnn(model_path=model_path, model_params=model_params)
        break
        file_paths, image_obj_list, real_labels = prepare_csv_data(args.test_csv)


        pred_labels, true_labels = get_predictions(model=model, input_data=(file_paths, image_obj_list, real_labels),
                                                   img_resolution=(w, h), classes=classes, conf_thrs=args.conf_thrs)
        print_confusion_matrix(y_pred_labels=pred_labels, y_true_labels=true_labels, classes=classes)
        print("\n\n")




def main_csv():
    parser = argparse.ArgumentParser(description='Test ResNet-18 on custom dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('test_csv', type=str, help='Path to the testing data directory')
    parser.add_argument('--model-path', type=str, required=True, help='path of the model')
    parser.add_argument('-pred-type', '--prediction-type', default="classification", type=str, help='path to the model')
    parser.add_argument('--preprocess', type=str, default="", help='preprocess options: Example Flip,GS')
    parser.add_argument('--conf-thrs', dest="conf_thrs", default=0.75, type=float, help='confidence threshold, -1 means no threshold')
    parser.add_argument('--print-info', action='store_true', help='path of the model')

    args = parser.parse_args()

    model_params = open(args.model_path[:-4] + ".txt").readlines()
    # model,(w, h),classes = load_model.load_model_cnn(model_path=args.model_path, model_params=model_params)
    # model, (w, h), classes = load_model.load_model_transformer(model_path=args.model_path, model_params=model_params)
    model, (w, h), classes = load_model.load_model_weights(model_path=args.model_path, model_params=model_params)

    file_paths,image_obj_list,real_labels = prepare_csv_data(args.test_csv)

    if args.conf_thrs==-1:
       args.conf_thrs=None

    pred_labels,true_labels = get_predictions(model=model, input_data=(file_paths,image_obj_list,real_labels),
                                              img_resolution=(w,h), classes=classes,  conf_thrs=args.conf_thrs)
    print_confusion_matrix(y_pred_labels=pred_labels, y_true_labels=true_labels, classes=classes)




def crawl_directory(main_folder, filters=(".jpg",".png")):
    import os
    file_paths = []

    # Walk through the directory and collect all file paths
    for root, dirs, files in os.walk(main_folder):
        for file in files:
          for filter in filters:
            if file.endswith((filter)):
              file_path = os.path.join(root, file)
              file_paths.append(file_path)
              break
            # if file.endswith((filter)):  # Adjust extensions as needed
            #     file_path = os.path.join(root, file)


    return file_paths

def main_dir():
    parser = argparse.ArgumentParser(description='Test ResNet-18 on custom dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test-dir', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--model-path', type=str, required=True, help='path of the model')
    parser.add_argument('-pred-type', '--prediction-type', default="classification", type=str, help='path to the model')
    parser.add_argument('--the-class', dest="label", default="", type=str, help='class name if there is only one class')
    parser.add_argument('--preprocess', type=str, default="None", help='preprocess options: Example Flip,GS')
    parser.add_argument('--conf-thrs', dest="conf_thrs", default=0.75, type=float, help='confience threshold, -1 means no threshold')

    args = parser.parse_args()

    model_params = open(args.model_path[:-4] + ".txt").readlines()
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

    print(properties)
    adp_pool = properties["adaptive pool output"]
    num_classes = properties["number of classes"]
    classes = properties["classes"]
    w, h = list(map(int, properties["resolution"].split("x")))
    # weight, height = list(map(int, args.res.split("x")))
    print("w,h", w, h)

    mymodel = CustomNet(num_classes=num_classes, adaptive_pool_output=adp_pool)

    # mymodel.load_state_dict(torch.load(args.model_path))
    try:
        checkpoint = torch.load(args.model_path)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
    except:
        mymodel.load_state_dict(torch.load(args.model_path))

    mymodel.eval()
    torch.no_grad()

    # files = fileio.getAllFilePaths(args.test_dir)
    files = crawl_directory(args.test_dir, filters=(".png",".jpeg", ".jpg"))
    files = [f for f in files if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
    cu.print_list(files, expl="files")
    # exit()
    image_obj_list = [Image.open(f) for f in files]
    if "Flip".lower() in args.preprocess.lower():
       image_obj_list = [img_obj.transpose(Image.FLIP_LEFT_RIGHT) for img_obj in image_obj_list]
    if "GS".lower() in args.preprocess.lower():
       image_obj_list = [pil_grayscale(img_obj) for img_obj in image_obj_list]

    # min_thrs = 0.75
    # min_thrs = None
    if args.conf_thrs==-1:
       args.conf_thrs=None

    results = [predict_image(img_obj, img_filename=img_file_name, model=mymodel,
                             res=(w, h), target_label=args.label,
                             labels=classes, min_prob_threshold=args.conf_thrs,
                             print_info=True)
               for img_file_name,img_obj in zip(files, image_obj_list)]

    confidences = [r[2] for r in results]
    judgements = [r[3] for r in results]
    labels = [r[1] for r in results]


    results = [(label,conf,score,os.path.basename(img_file)) for (label,conf,score,img_file) in
                zip(labels,confidences,judgements,files) if not score is None]


    results.sort(key=lambda X:X[2])
    cu.print_list(results)
    # print("labels", labels)
    if not args.label == "":
       correct_number = len([(l,c,s,filename) for (l,c,s,filename) in results if l == args.label and (s is not None)])
       print("len labels", len(results), "correct:", correct_number)
       acc = correct_number / len(results)
       print("accuracy %"+str(acc*100))

    # if not args.label == "":
    #    correct_number =  len([l for l in labels if l==args.label])
    #    print("len labels", len(labels), "correct:", correct_number)
    #    acc = correct_number/len(labels)
    #    print("accuracy %"+str(acc*100))

def pil_grayscale(image_rgb_obj):
    image_gs = ImageOps.grayscale(image_rgb_obj)
    rgbimg = Image.merge("RGB", (image_gs, image_gs, image_gs))
    return rgbimg


def main_single_image():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on custom dataset')
    parser.add_argument('--img', type=str, required=True, help='path to the image file')
    parser.add_argument('--model-path', type=str, required=True, help='path to the model')
    parser.add_argument("--res", default="128x128",
                         help="Target size for resizing in the format 'wxh' (default: 128x128).")
    parser.add_argument('--preprocess', type=str, default="None", help='preprocess options: Example Flip,GS')

    parser.add_argument('-pred-type', '--prediction-type', default="classification", type=str, help='path to the model')


    args = parser.parse_args()

    model_params = open(args.model_path[:-4]+".txt").readlines()
    print("model_params", model_params)

    properties = load_model.load_model_parameters(model_params)

    print(properties)


    model, (w, h), classes = load_model.load_model_weights(model_path=args.model_path, model_params=model_params)

    img = cv2.imread(args.img, 1)


    if "regress" in args.prediction_type:
       output = predict_regress(args.img, model, res=(w, h))
       print("output ", output)
       # img = cv2.resize(img, (weight, height))
       img_copy = np.copy(img)

       if len(output) == 2: #we assume center
          h,w = img.shape[:2]
          x = int(w * output[0])
          y = int(h * output[1])
          print("x", x)
          print("y", y)
          cv2.circle(img, center=(x,y), radius=5, color=(200,20,200), thickness=10)
          cv2.rectangle(img, pt1=(x-40,y-40), pt2=(x+40,y+40), color=(200,20,200))

       if len(output) == 3:  # we assume size center
           h, w = img.shape[:2]
           print("w",w,"h",h)

           #output = [0.7, 0.8, 0.3]
           output = [1.0, 0.2, 0.4]

           print("output[0]", output[0], "output[0]/5", output[0]/5)
           print("w*h", w*h)
           radius = (output[0]/5)*((w*h)**0.5)
           print ("rad:", radius)
           x = int(w * output[1])
           y = int(h * output[2])
           print("x", x)
           print("y", y)
           # radius = 12017**0.5
           print("radius", radius)
           cv2.circle(img, center=(x, y), radius=5, color=(200, 20, 200), thickness=10)
           # cv2.rectangle(img, pt1=(x - 40, y - 40), pt2=(x + 40, y + 40), color=(200, 20, 200))
           cv2.rectangle(img, pt1=(x - int(radius/2), y - int(radius/2)), pt2=(x + int(radius/2), y + int(radius/2)), color=(200, 20, 200))

       if len(output) == 4:
           h, w = img.shape[:2]

           # output = [0.3,0.1,0.7,0.5]

           W,H,xc,yc = output

           xc = int(w * xc)
           yc = int(h * yc)


           W = int(W * w)
           H = int(H * h)

           # print("output:", output)
           cv2.circle(img, center=(xc, yc), radius=5, color=(200, 20, 200), thickness=10)
           cv2.rectangle(img, pt1=(xc - int(W / 2), yc - int(H / 2)),
                              pt2=(xc + int(W / 2), yc + int(H / 2)), color=(200, 20, 200))


       cv2.imshow("det", img)
       # cv2.imshow("org", img_copy)
    else:
       img_obj = Image.open(args.img)

       if not args.preprocess == "None":
          if "Flip".lower() in args.preprocess.lower():
              img_obj = img_obj.transpose(Image.FLIP_LEFT_RIGHT)
              img = cv2.flip(img, 1)
          if "GS".lower() in args.preprocess.lower():
              img_obj = pil_grayscale(img_obj)
              img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

       pred, label, conf, filepath = predict_image(img_obj, img_filename="", model=model, labels=classes, res=(w, h))
       print(pred, type(pred), label, classes)
       cv2.imshow("class " + label+" prob"+str(round(conf,2)), img)

    cv2.waitKey(0)

def predict_image_filepath(filepath, model, labels=("acura", "alpha romeo"), res=(128,128), min_prob_threshold=0.75):
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    import torchvision
    from scipy.special import softmax

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is", device)
    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)
    image = Image.open(filepath)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    model.to(device=device)
    image = image.resize(res)
    data = TF.to_tensor(image)
    data = normalizer(data)
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)
    print("output", output.cpu().detach().numpy()[0])
    _, predicted = torch.max(output.data, 1)
    predicted_numpy = predicted.cpu().detach().numpy()

    # clamped_outputs = output.clamp(0, 1)

    raw_output = output.cpu().detach().numpy()
    # min_val = np.min(raw_output)
    # max_val = np.max(raw_output)
    # raw_output_norm = (raw_output - min_val)/max_val

    # probabilities = F.softmax(output, dim=0)
    probabilities = np.round(softmax(raw_output), 2)
    confidence_value = np.max(probabilities)
    # print("probs:::",probabilities)
    # from sklearn import
    # raw_output_norm = clamped_outputs.cpu().detach().numpy()
    # print("raw output", raw_output, "pred:", predicted_numpy, labels[predicted])
    final_predicted_value = labels[predicted]
    if min_prob_threshold > confidence_value:
        final_predicted_value = "unsure"
        pass
    else:
        print(os.path.basename(filepath), end=" ")
        print("prob", probabilities, "confidence:", confidence_value, "pred:", predicted_numpy, final_predicted_value)
    return predicted_numpy, labels[predicted], confidence_value  # this part is used for the single main

def predict_image(image_obj, img_filename, model, labels,
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

    if min_prob_threshold == None:
       if print_info:
           print ("prob",probabilities,"confidence:",confidence_value, "pred:",predicted_numpy,"real pred val",
               final_predicted_value, "Target:", target_label,  os.path.basename(img_filename))
       score = True if final_predicted_value==target_label else False
    else:
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

def predict_regress(filepath, model, res=(128,128)):
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    import torchvision
    from PIL import Image
    from scipy.special import softmax

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)
    image = Image.open(filepath)
    model.to(device=device)
    image = image.resize(res)
    data = TF.to_tensor(image)
    data = normalizer(data)
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)

    return output.cpu().detach().numpy()[0]

import sys

if __name__ == '__main__':
    commands = ['single','dir', 'csv', 'csv-multi-model']

    if len(sys.argv)==1:
       print ('options are',commands)
       exit()

    inputCommand = sys.argv[1]
    commandArgs = sys.argv[1:]

    print("inputCommand, commandArgs:", inputCommand, commandArgs)

    if sys.argv[1] == commands[0]:
       sys.argv = commandArgs
       main_single_image()
    elif sys.argv[1]==commands[1]:
       sys.argv = commandArgs
       main_dir()
    elif sys.argv[1]==commands[2]:
       sys.argv = commandArgs
       main_csv()
    elif sys.argv[1] == commands[3]:
       sys.argv = commandArgs
       main_csv_multi_model()

