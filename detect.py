import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_minimal import (
    YoloV3,
    transform_images, load_tfrecord_dataset, draw_outputs
)

flags.DEFINE_string('weights', '', 'path to weights file')
flags.DEFINE_integer('size', 256, 'resize images to')
flags.DEFINE_string('tfrecord', './tfrecord-data/axial_brain_val.tfrecord', 'path to tfrecord file')
flags.DEFINE_string('output', './output-imgs', 'path to output images directory')

def main(_argv):

    yolo = YoloV3(size=FLAGS.size, classes=2)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = ['negative', 'positive']

    dataset = load_tfrecord_dataset(FLAGS.tfrecord)

    for input_idx, (img_raw, _label) in enumerate(dataset):

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('inference time: {}'.format(t2 - t1))

        if nums[0]:

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))

            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            output_filename = f"{FLAGS.output}/{input_idx}.jpg"
            cv2.imwrite(output_filename, img)
            logging.info('output saved to: {}'.format(output_filename))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
