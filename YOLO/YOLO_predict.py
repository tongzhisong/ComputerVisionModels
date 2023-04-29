import tensorflow as tf
import numpy as np
import os
from matplotlib.pyplot import imshow
from PIL import Image
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import scale_boxes, read_classes, read_anchors, preprocess_image, get_colors_for_classes, draw_boxes

class_names = read_classes("coco_classes.txt")
anchors = read_anchors("yolo_anchors.txt")
yolo2_model = load_model("model")

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):

    # (19, 19, 5, 80)
    box_scores = box_confidence * box_class_probs
    # (19, 19, 5)
    box_classes = tf.math.argmax(box_scores, axis=-1)
    # (19, 19, 5)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    # (19, 19, 5)
    filtering_mask = np.array(box_class_scores >= threshold)
    # (19, 19, 5) to (None,)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    # (19, 19, 5, 4) to (None, 4)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    # (19, 19, 5) to (None,)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

def iou(box1, box2):

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    inter_area = max(inter_width, 0) * max(inter_height, 0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area/union_area

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):

    # scalar integer tensor
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):

    # (19, 19, 5, 2)
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    # (y_min, x_min, y_max, x_max)
    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],
        box_mins[..., 0:1],
        box_maxes[..., 1:2],
        box_maxes[..., 0:1]
    ])

def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)

    return scores, boxes, classes

def predict(image_file):

    # image: a python (PIL) representation of your image used fro drawing boxes.
    # image_data: a numpy-array representing the image. Will be the input to the CNN.
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    yolo_model_outputs = yolo2_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5)

    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    image.save(os.path.join("out", image_file), quality=100)
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict("test.jpg")

