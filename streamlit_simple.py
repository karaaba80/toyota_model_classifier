import os.path

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

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

    img_obj = None

    if not selected_img_obj == None:
       show_image(selected_img_obj, image_placeholder)
       #image_placeholder.image(selected_img_obj, caption=caption, use_column_width=True)
       # img_obj = selected_img_obj

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if not uploaded_file == None:
       img_obj = Image.open(uploaded_file)
       # image_placeholder.image(img_obj, caption='Car Not Found', use_column_width=True)

    image_url = st.text_input("Or enter an image URL")
    if st.button("Load Image from URL") and not image_url == "":  # Only trigger on button click
       img_obj = Image.open(BytesIO(requests.get(image_url).content))
       # image_placeholder.image(img_obj, caption='Car Not Found', use_column_width=True)

    # if not img_obj == None:
    if selected_img_obj == None and not img_obj == None:
       #image_placeholder.image(img_obj, caption='', use_column_width=True)
       show_image(img_obj, image_placeholder)


def show_image(img_obj, image_placeholder):
    image_placeholder.image(img_obj, caption='', use_column_width=True)

if __name__ == "__main__":
    main()