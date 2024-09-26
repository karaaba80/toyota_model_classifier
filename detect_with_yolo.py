import sys
import torch
import numpy
import cv2

def yolo_detect_bbox(img_obj, min_wh2image_ratio=0.2, min_obj_conf=0.9):
    """
    ----------
    Author: M. Faik Karaaba (karaaba80), modified from Damon Gwin
    ----------
    - bbox detector using Yolo
    ----------
    """
    sys.path.insert(0, "YoloV3")

    from YoloV3.utilities.configs import parse_config, parse_names #Author: Damon Gwinn (gwinndr)
    from YoloV3.utilities.weights import load_weights #Author: Damon Gwinn (gwinndr)

    from YoloV3.utilities.devices import gpu_device_name, get_device, use_cuda #Author: Damon Gwinn (gwinndr)
    from YoloV3.utilities.images import load_image, get_bboxes_xywh_with_class_filters #Author: Damon Gwinn (gwinndr)
    from YoloV3.utilities.inferencing import inference_on_image, inference_video_to_video #Author: Damon Gwinn (gwinndr)

    cfg = "./YoloV3/configs/yolov3.cfg"
    class_names = "./YoloV3/configs/coco.names"
    weights = "./YoloV3/weights/yolov3.weights"


    # no_grad disables autograd so our model runs faster
    with torch.no_grad():
        print("Parsing config into model...")
        model = parse_config(cfg)
        if (model is None):
            return

        model = model.to(get_device())
        model.eval()

        print("yolo model is loaded...")

        # Network input dim
        if (model.net_block.width != model.net_block.height):
            print("Error: Width and height must match in [net]")
            return

        network_dim = model.net_block.width

        # Letterboxing
        letterbox = True #preserve the aspect ratio

        print("Parsing class names...")
        class_names = parse_names(class_names)
        if (class_names is None):
            return

        print("Loading weights...")
        load_weights(model, weights)

        try:
            image = cv2.cvtColor(numpy.array(img_obj), cv2.COLOR_RGB2BGR)
            # image = numpy.array(img_obj.getdata()).reshape(img_obj.size[0], img_obj.size[1], 3)
        except Exception as E:
            print("exception", E)

        # image = load_image(img_path)
        if (image is None):
            print("image is none")
            return

        im_h, im_w = image.shape[:2]
        detections = inference_on_image(model, image, network_dim, obj_thresh=min_obj_conf, letterbox=letterbox)
        bboxess_probs = get_bboxes_xywh_with_class_filters(detections, class_names, class_filters=["car"]) #we want only "car" objects

        print(bboxess_probs)

    min_w = min_wh2image_ratio * im_w  # minimum w of the bbox
    min_h = min_wh2image_ratio * im_h  # minimum h of the bbox

    bboxess_probs = [((x, y, w, h), probability) for (x, y, w, h), probability in bboxess_probs if w >= min_w and h > min_h]

    return bboxess_probs

def convert_bbox_xywh2xyxy(bbox_xywh):
    """
    ----------
    Author: M. Faik Karaaba (karaaba80)
    ----------
    """

    x1, y1, w, h = bbox_xywh #x,y,w,h
    bbox_xyxy = x1, y1, x1 + w, y1 + h
    return bbox_xyxy