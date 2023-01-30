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
flags.DEFINE_string('tfrecord', './tfrecord-data/axial-_brain_val.tfrecord', 'path to tfrecord file')
flags.DEFINE_string('output', './output.jpg', 'path to output image')

def main(_argv):

    yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = ['negative', 'positive']

    dataset = load_tfrecord_dataset(
        FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
    dataset = dataset.shuffle(512)
    img_raw, _label = next(iter(dataset.take(1)))

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
