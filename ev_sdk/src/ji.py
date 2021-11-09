from __future__ import print_function

import logging as log
import json
import os
import tensorflow as tf
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

project_root = '/project/train/src_repo'
sys.path.append(os.path.join(project_root, 'tf-models/research'))
sys.path.append(os.path.join(project_root, 'tf-models/research/slim'))

from object_detection.utils import visualization_utils as viz_utils

def plot_save_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys 1n the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  print(image_np_with_annotations.shape)
  img_result = viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=False,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, img_result)
    print("saved done.")

log.basicConfig(level=log.DEBUG)

sess = None
input_w, input_h, input_c, input_n = (300, 300, 3, 1)

# Replace your own target label here
label_id_map = {
    1: "phone",
    2: "person"
}


def init():
    """Initialize model

    Returns: model

    """
    model_pb_path = "/project/train/models/final/ssd_inception_v2_detetion.pb"
    if not os.path.isfile(model_pb_path):
        log.error(f'{model_pb_path} does not exist')
        return None
    log.info('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    log.info('Initializing session...')
    global sess
    sess = tf.Session(graph=detection_graph)
    return detection_graph


def process_image(net, input_image, args=None):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: optional args

    Returns: process result

    """

    # ------------------------------- Prepare input -------------------------------------
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    ih, iw, _ = input_image.shape
    show_image = input_image
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
        
        
    input_image = np.expand_dims(input_image, axis=0)
    
    # --------------------------- Performing inference ----------------------------------
    # Extract image tensor
    image_tensor = net.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes, scores, classes, number of detections
    boxes = net.get_tensor_by_name('detection_boxes:0')
    scores = net.get_tensor_by_name('detection_scores:0')
    classes = net.get_tensor_by_name('detection_classes:0')
    num_detections = net.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: input_image})

    # --------------------------- Read and postprocess output ---------------------------
    scores = np.squeeze(scores)
    valid_index = len(scores[scores >= 0.5])

    boxes = np.squeeze(boxes)[:valid_index]
    boxes[:, 0] *= ih
    boxes[:, 2] *= ih
    boxes[:, 1] *= iw
    boxes[:, 3] *= iw
    boxes = boxes.astype(np.int32)
    classes = np.squeeze(classes)[:valid_index]
    scores = scores[:valid_index]
    '''
    phone_class_id = 1
    person_class_id = 1
    category_index = {int(phone_class_id): {'id': 1, 'name': 'phone'}, int{person_class_id}:{'id':1, 'name': 'person'}}
    plot_save_detections(show_image,
                        boxes,
                        classes.astype(int),
                        scores,
                        category_index,
                        image_name = "test_image.jpg")
    '''
    
    result = {'objects': []}
    for k, score in enumerate(scores):
        label = np.int(classes[k])
        if label not in label_id_map:
            log.warning(f'{label} does not in {label_id_map}')
            continue
        ymin, xmin, ymax, xmax = boxes[k]
        result['objects'].append({
            'name': label_id_map[label],
            "confidence":float(score),
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
        })
    return json.dumps(result, indent = 4)  


if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/home/data/334/phone_10367.jpg')
    predictor = init()
    result = process_image(predictor, img)
    ##log.info(result)
    
