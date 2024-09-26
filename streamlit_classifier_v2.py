import os.path

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import numpy as np
import cv2
import sys
import load_model
import detect_with_yolo
import main_utils

sys.path.insert(0, "YoloV3")

from YoloV3.utilities.configs import parse_config, parse_names  # Author: Damon Gwinn (gwinndr)
from YoloV3.utilities.weights import load_weights  # Author: Damon Gwinn (gwinndr)

from YoloV3.utilities.devices import gpu_device_name, get_device, use_cuda  # Author: Damon Gwinn (gwinndr)
from YoloV3.utilities.images import load_image, get_bboxes_xywh_with_class_filters  # Author: Damon Gwinn (gwinndr)
from YoloV3.utilities.inferencing import inference_on_image

def create_thumbnails_(example_images):
    for example_name, image_path in example_images.items():
        if os.path.exists(image_path):
           img = Image.open(image_path)
           st.sidebar.image(img, caption=example_name, width=90)  # Adjust width for thumbnails

def create_thumbnails_and_get_selected_image(example_images):
    # Create columns for thumbnails
    cols = st.sidebar.columns(len(example_images) - 1)  # Exclude the "No Car Selected" option

    selected_img_obj = None
    image_caption = ""
    # Iterate through the example images, excluding the first option
    for i, (key, img_path) in enumerate(list(example_images.items())[1:], start=1):
        with cols[i-1]:
            img_obj = Image.open(img_path)
            st.image(img_obj, caption=key, width=100)  # Display thumbnail

            # Create a button under each thumbnail
            # if st.button(f"Select {key}"):
            if st.button(key):
               # image_placeholder.image(img, caption=f"Selected: {key}", use_column_width=True)
               selected_img_obj = img_obj
               image_caption = os.path.basename(os.path.splitext(img_path)[0])
               # print("key",key)
    # print("create_thumbnails")
    return selected_img_obj,image_caption

@st.cache_resource
def load_model_local():
    model_path = "classifier_weights/Transformer_Latest.pth"
    model_params = open(model_path[:-4] + ".txt").readlines()
    model, (img_width, img_height), classes = load_model.load_model_weights(model_path=model_path,
                                                                            model_params=model_params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() #necessary to disable any drop out units and further
    torch.no_grad()

    resolution = img_width, img_height

    return model,resolution,classes

def crop_image_obj_bbox(img_obj, bboxes_probs):
    img_obj_crop = None
    img_obj_with_rect_bbox = None
    if len(bboxes_probs) > 0:
       (xx, yy, ww, hh), conf_detection = bboxes_probs[0]
       x1, y1, x2, y2 = detect_with_yolo.convert_bbox_xywh2xyxy((xx, yy, ww, hh))
       img_cv2 = main_utils.pil2cv2(img_obj)
       cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)
       img_obj_with_rect_bbox = main_utils.cv2topil(img_cv2)
       # print("bboxes_probs", bboxes_probs)
       # (img_width, img_height) = img_resolution
       img_obj_crop = img_obj.crop((x1, y1, x2, y2))
    return img_obj_crop, img_obj_with_rect_bbox

@st.cache_resource
def load_yolo_once():
    import sys

    #Author: M. Faik Karaaba (karaaba80), modified from Damon Gwin    
    #bbox detector using Yolo



    cfg = "./YoloV3/configs/yolov3.cfg"
    class_names = "./YoloV3/configs/coco.names"
    weights = "./YoloV3/weights/yolov3.weights"

    # no_grad disables autograd so our model runs faster
    with torch.no_grad():
        print("YOLO Parsing config into model...")
        yolo_model = parse_config(cfg)
        if (yolo_model is None):
            return

        yolo_model = yolo_model.to(get_device())
        yolo_model.eval()

        print("yolo model is loaded...")

        # Network input dim
        if (yolo_model.net_block.width != yolo_model.net_block.height):
            print("Error: Width and height must match in [net]")
            return

        # Letterboxing
        letterbox = True #preserve the aspect ratio

        print("Parsing class names...")
        class_names = parse_names(class_names)
        if (class_names is None):
            return

        print("Loading weights...")
        load_weights(yolo_model, weights)

    return yolo_model,class_names

def predict_image_object(*, img_object, model, labels, res=(128,128), min_prob_threshold=0.75):
    # print("filepath", filepath, "\n")
    # print("predict_image_object is loaded")
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    import torchvision
    from PIL import Image
    from scipy.special import softmax

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)

    image = img_object
    # print("image", image.size)
    model.to(device=device)
    image = image.resize(res)
    # image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
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
    # min_val = np.min(raw_output)
    # max_val = np.max(raw_output)
    # raw_output_norm = (raw_output - min_val)/max_val

    # probabilities = F.softmax(output, dim=0)
    probabilities = np.round(softmax(raw_output), 2)
    confidence_value = np.max(probabilities)

    final_predicted_value = labels[predicted]
    if min_prob_threshold > confidence_value:
        final_predicted_value = "unsure"
        pass
    # else:
        # print(os.path.basename(filepath), end=" ")
        # print("prob", probabilities, "confidence:", confidence_value, "pred:", predicted_numpy, final_predicted_value)
    return predicted_numpy, labels[predicted], confidence_value  # this part is used for the single main

# Load the model once
model,resolution,classes = load_model_local()
yolo_model,yolo_class_names = load_yolo_once()

def get_bboxes_yolo(yolo_model, image_obj, min_wh2image_ratio=0.2, min_obj_conf=0.9):
    image = None
    try:
        image = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2BGR)
        # image = numpy.array(img_obj.getdata()).reshape(img_obj.size[0], img_obj.size[1], 3)
    except Exception as E:
        print("exception", E)

    network_dim = yolo_model.net_block.width
    detections = inference_on_image(yolo_model, image, network_dim, obj_thresh=min_obj_conf, letterbox=True)
    bboxess_probs = get_bboxes_xywh_with_class_filters(detections, yolo_class_names,
                                                       class_filters=["car"])  # we want only "car" obj

    im_h,im_w = image.shape[:2]
    min_w = min_wh2image_ratio * im_w  # minimum w of the bbox
    min_h = min_wh2image_ratio * im_h  # minimum h of the bbox

    bboxess_probs = [((x, y, w, h), probability) for (x, y, w, h), probability in bboxess_probs if
                     w >= min_w and h > min_h]

    return bboxess_probs

def detect_predict_and_show(img_obj, image_placeholder):
    bboxes_probs = get_bboxes_yolo(yolo_model, img_obj)
    img_obj_crop, img_obj_with_rect_bbox = crop_image_obj_bbox(img_obj, bboxes_probs)
    # print("bboxes_probs", bboxes_probs)
    np_value, label, conf = predict_image_object(img_object=img_obj_crop, model=model,
                                                 labels=classes, res=resolution)
    image_placeholder.image(img_obj_with_rect_bbox, caption=label, use_column_width=True)

def main():
    # keys = ["No Car Selected", "Car "+i, "Car 2"]
    images = ["./examples/"+f for f in os.listdir("./examples/")]
    keys = ["No Car Selected"]
    keys.extend(["car "+str(i) for i in range(len(images))])
    example_images = {keys[0] : ""}
    for i,im_path in enumerate(images):
        example_images["car"+str(i)] = im_path

    st.sidebar.header("Select an example image")
    st.title("Aygo Yaris Auris")
    st.header("Toyota Model Classifier")
    image_placeholder = st.empty()

    selected_img_obj,caption = create_thumbnails_and_get_selected_image(example_images)
    # selected_example = st.sidebar.selectbox("Example images", list(example_images.keys()), index=0)
    print("type of selected img:",type(selected_img_obj))

    if not selected_img_obj == None:
       detect_predict_and_show(selected_img_obj, image_placeholder)

    img_obj = None
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if not uploaded_file == None:
       img_obj = Image.open(uploaded_file)
       # image_placeholder.image(img_obj, caption='Car Not Found', use_column_width=True)

    image_url = st.text_input("Or enter an image URL")
    print("image url:", image_url)
    button_clicked = st.button("Load Image from URL")
    print("for URL button clicked?", button_clicked)
    if button_clicked and not image_url == "":  # Only trigger on button click
       img_obj = Image.open(BytesIO(requests.get(image_url).content))


    # if not img_obj == None:
    if selected_img_obj == None and not img_obj == None:
       # image_placeholder.image(img_obj, caption=label, use_column_width=True)
       detect_predict_and_show(img_obj, image_placeholder)

if __name__ == "__main__":
    main()