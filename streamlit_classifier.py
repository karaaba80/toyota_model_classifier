import os.path

import streamlit as st
import requests
from io import BytesIO

from PIL import Image, ImageOps
import numpy as np

import torch
import cv2

from networks import CNN_Net
import load_model
import detect_with_yolo
import main_utils

def predict_image_object(*, img_object, model, labels, res=(128,128), min_prob_threshold=0.75):
    # print("filepath", filepath, "\n")
    print("predict_image_object is loaded")
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    import torchvision
    from PIL import Image
    from scipy.special import softmax

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalizer = torchvision.transforms.Normalize(mean=0.5, std=0.5)

    image = img_object
    print("image", image.size)
    model.to(device=device)
    image = image.resize(res)
    # image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
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

    final_predicted_value = labels[predicted]
    if min_prob_threshold > confidence_value:
        final_predicted_value = "unsure"
        pass
    else:
        # print(os.path.basename(filepath), end=" ")
        print("prob", probabilities, "confidence:", confidence_value, "pred:", predicted_numpy, final_predicted_value)
    return predicted_numpy, labels[predicted], confidence_value  # this part is used for the single main

def read_model_properties(model_params_path):
    model_params = open(model_params_path).readlines()
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
    resolution = list(map(int, properties["resolution"].split("x")))

    return adp_pool,num_classes,classes,resolution

def add_combos_in_a_row(text_for_combo1='Select option for Combo Box 1',
                        text_for_combo2='Select option for Combo Box 2',
                        options1=('Option 11', 'Option 2', 'Option 3'),
                        options2=('Option AA', 'Option BB', 'Option CC')):
    # Create two columns
    col1, col2 = st.columns(2)

    # Add a combo box to each column
    with col1:
        option1 = st.selectbox(text_for_combo1, options1)
        st.write(f'You selected: {option1}')

    with col2:
        option2 = st.selectbox(text_for_combo2, options2)
        st.write(f'You selected: {option2}')

    return option1,option2

def pil_grayscale(image_rgb_obj):
    image_gs = ImageOps.grayscale(image_rgb_obj)
    rgbimg = Image.merge("RGB", (image_gs, image_gs, image_gs))
    return rgbimg

def predict_and_show(img_obj, model, classes, img_resolution, image_placeholder):
    print("hello")
    bboxes_probs = detect_with_yolo.yolo_detect_bbox(img_obj=img_obj, min_wh2image_ratio=0.2)

    if len(bboxes_probs) > 0:
       (xx, yy, ww, hh), conf_detection = bboxes_probs[0]
       x1, y1, x2, y2 = detect_with_yolo.convert_bbox_xywh2xyxy((xx, yy, ww, hh))
       img_cv2 = main_utils.pil2cv2(img_obj)
       cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)
       img_obj_with_bbox = main_utils.cv2topil(img_cv2)
       print("bboxes_probs", bboxes_probs)
       (img_width, img_height) = img_resolution
       img_obj_crop = img_obj.crop((x1, y1, x2, y2))
       np_value, label, conf = predict_image_object(img_object=img_obj_crop, model=model, labels=classes,
                                                 res=(img_width, img_height))
       image_placeholder.image(img_obj_with_bbox, caption='Model:' + label + " Confidence:" + str(round(conf, 2)),
                            use_column_width=True)
    else:
       image_placeholder.image(img_obj, caption='Car Not Found', use_column_width=True)

def main():
    keys = ["No Car Selected", "Car 1", "Car 2"]
    # List of example images
    example_images = {
        keys[0] : "",
        keys[1] : "./examples/auris.jpg",
        keys[2] : "examples/toyota-yaris-2009.jpg",
        # "Car 3": "examples/car3.jpg",
    }

    # Create thumbnails in the sidebar
    for example_name, image_path in example_images.items():
        if not example_name==keys[0]:
           img = Image.open(image_path)
           st.sidebar.image(img, caption=example_name, width=90)  # Adjust width for thumbnails

    if 'selected_source' not in st.session_state:
        st.session_state.selected_source = ""

    # Sidebar for selecting example images
    st.sidebar.header("Select an example image")
    # options = ["Select an example image",""] + list(example_images.keys())
    selected_example = st.sidebar.selectbox("Example images", list(example_images.keys()), index=0)
    # selected_example = st.sidebar.selectbox("Example images", options)


    print("selected_example",selected_example)
    # Load and display the selected image
    example_img_path = ""
    if not selected_example==keys[0]:
       st.session_state.selected_source = "example"
       print("only if a car here","selected_example",selected_example)
       example_img_path = example_images[selected_example]
    #     # print("img_path",img_path, os.path.exists(img_path),example_images["Car 1"]==img_path)
    #    img = Image.open(example_img_path)
    #    st.image(img, caption=selected_example, use_column_width=True)

    image_url = ""


    # Title
    st.title("Aygo Yaris Auris")
    # Header
    st.header("Toyota Model Classifier")
    # Text


    # model_params_path = "classifier_weights/Transformer_Latest.txt"
    model_path = "classifier_weights/Transformer_Latest.pth"


    # adp_pool,num_classes,classes,resolution = read_model_properties(model_params_path)
    model_params = open(model_path[:-4] + ".txt").readlines()
    print("device",torch.device('cpu'))
    model, (img_width, img_height), classes = load_model.load_model_weights(model_path=model_path,
                                                                            model_params=model_params, device=torch.device('cuda'))

    # w,h = resolution
    # print("w,h", w, h)
    # print("classes", list(classes))

    # model = CNN_Net.ResNet(num_classes=num_classes, adaptive_pool_output=adp_pool)

    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.eval() #necessary to disable any drop out units and further
    # torch.no_grad()

    # Initialize session state
    if 'previous_url' not in st.session_state:
        st.session_state.previous_url = ''

    #file upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    print("uploaded_file", uploaded_file)
    if not uploaded_file == None:
       # selected_example = keys[0]
       st.session_state.selected_source = "upload"
    # URL input
    image_url = st.text_input("Or enter an image URL")

    print("st.session_state.previous_url", st.session_state.previous_url)
    print("image_url",image_url)
    if not image_url=="":
       st.session_state.selected_source = "url"

    # load_button_result = st.button("Load Image from URL")
    # result = st.button("Load Image from URL")
    # print("button result", result)
    print("st.session_state.selected_source",st.session_state.selected_source)
    # color_mode, flip_or_not = add_combos_in_a_row(text_for_combo1="choose color", options1=("RGB", "Grayscale"),
    #                                               text_for_combo2="choose flip", options2=("Org", "Flip"))
    print("uploaded_file::",uploaded_file)
    image_org = None
    image_placeholder = st.empty()
    print("selected_example",selected_example)
    try:
        if not selected_example==keys[0] and st.session_state.selected_source=="example":
            image_org = Image.open(example_img_path)
            predict_and_show(img_obj=image_org, model=model, classes=classes,
                             img_resolution=(img_width, img_height),
                             image_placeholder=image_placeholder)
        elif not image_url == "" and st.session_state.selected_source=="url":
           # selected_example
           image_org = Image.open(BytesIO(requests.get(image_url).content))
           predict_and_show(img_obj=image_org, model=model, classes=classes,
                             img_resolution=(img_width, img_height),
                             image_placeholder=image_placeholder)
           image_url = ""
        else: #this is with url
            # print("here...")
            image_org = Image.open(uploaded_file)
            predict_and_show(img_obj=image_org, model=model, classes=classes,
                             img_resolution=(img_width, img_height),
                             image_placeholder=image_placeholder)
            uploaded_file = None

    except Exception as E:
            print("exception", E)
            pass


if __name__ == '__main__':
    main()
