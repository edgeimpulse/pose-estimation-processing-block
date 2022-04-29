import numpy as np
import cv2, io, base64
from PIL import Image
# Import TF and TF Hub libraries.
import tensorflow as tf
# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, high_threshold, low_threshold, gaussian_blur):
    # features is a 1D array, reshape so we have a matrix
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    graphs = []
    all_features = []

    width = raw_data[0][0]
    height = raw_data[1][0]
    raw_data = raw_data[2:].astype(dtype=np.uint32).view(dtype=np.uint8)

    pixels_per_frame = height * width * 4
    frame_count = 0
    expected_frame_count = len(raw_data) / pixels_per_frame

    for i in np.arange(0, len(raw_data), pixels_per_frame):
        # grab the image (there could be multiple potentially)
        frame = (raw_data.flatten()[i:i+pixels_per_frame]).reshape((height, width, 4))[:, :, :3]

        # Load the input image.
        # image = tf.convert_to_tensor(frame)
        # image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        # image = tf.image.resize_with_pad(image, 192, 192)

        # convert into an image
        # img = Image.fromarray(frame, mode='RGBA')
        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

        # https://www.tensorflow.org/hub/tutorials/movenet

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
            input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
            """
            # TF Lite format expects tensor type of uint8.
            input_image = tf.cast(input_image, dtype=tf.uint8)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
            # Invoke inference.
            interpreter.invoke()
            # Get the model prediction.
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            return keypoints_with_scores


        # Output is a [1, 1, 17, 3] numpy array.
        keypoints_with_scores = movenet(np.expand_dims(frame, axis=0))


        # convert to grayscale
        # img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        # # blur the image for better edge detection
        # img_blur = cv2.GaussianBlur(img_gray, (int(gaussian_blur), int(gaussian_blur)), 0)

        # edges = cv2.Canny(img_blur, float(high_threshold), float(low_threshold))

        if draw_graphs:
            output_overlay = draw_prediction_on_image(
                np.squeeze(frame), keypoints_with_scores)

            im = Image.fromarray(output_overlay).convert(mode='RGBA')
            buf = io.BytesIO()
            im.save(buf, format='PNG')

            buf.seek(0)
            image = (base64.b64encode(buf.getvalue()).decode('ascii'))

            buf.close()

            name = 'Image'
            if expected_frame_count > 1:
                name = 'Frame ' + str(frame_count)

            graphs.append({
                'name': name,
                'image': image,
                'imageMimeType': 'image/png',
                'type': 'image'
            })

    # all_features = all_features + [ float(x / 255) for x in edges.flatten().tolist() ]
    
    ### %%%SRH%%% Modify output to be flat "image" data to trick Studio
    all_features = keypoints_with_scores.flatten()
    return {
        'features': all_features,
        'graphs': graphs,
        'output_config': {
            'type': 'image',
            'shape': {
                'width': len(all_features),
                'height': int(1),
                'channels': 1
            }
        }
    }
    
    #return {
        # 'features': [ int(x) for x in all_features ],
    #    'features': keypoints_with_scores.flatten(),
    #    'graphs': graphs,
    #    'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
    #        'type': 'flat',
            # 'shape': {
            #     # shape should be { width, height, channels } for image, { width, height } for spectrogram
            #     'width': int(width),
            #     'height': int(height),
            #     'channels': 1
            # }
    #    }
    #}
    ### %%%END%%%

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  # %%%SRH%%% Convert image from RGB to BGR before displaying it
  im = ax.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
  #im = ax.imshow(image)
  # %%%END%%%
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot