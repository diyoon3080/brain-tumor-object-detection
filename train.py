from absl import app, flags
from absl.flags import FLAGS

import tensorflow as tf
import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_minimal import (
    YoloV3, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    freeze_all, load_tfrecord_dataset, transform_images, transform_targets
)

flags.DEFINE_string('orientation', '', 'orientation of the training data')
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

def setup_model():

    model = YoloV3(size=256, channels=3, training=True, classes=2)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    pretrained_model = YoloV3(size=256, training=True, classes=80)
    pretrained_model.load_weights(FLAGS.weights)

    model.get_layer('yolo_darknet').set_weights(pretrained_model.get_layer('yolo_darknet').get_weights())
    freeze_all(model.get_layer('yolo_darknet'))

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=2) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss, anchors, anchor_masks


def main(_argv):

    model, optimizer, loss, anchors, anchor_masks = setup_model()

    train_dataset = load_tfrecord_dataset(FLAGS.dataset)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)

    train_dataset = train_dataset.map(lambda x, y: (transform_images(x, size=256), transform_targets(y, anchors, anchor_masks, size=256)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = load_tfrecord_dataset(FLAGS.val_dataset)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (transform_images(x, size=256), transform_targets(y, anchors, anchor_masks, size=256)))

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=5, verbose=1, monitor='val_loss'),
        ModelCheckpoint(f'checkpoints/{FLAGS.orientation}_ckpt.tf', verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss'),
        TensorBoard(log_dir='logs')
    ]

    start_time = time.time()
    model.fit(train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=val_dataset)
    end_time = time.time() - start_time
    print(f'Total Training Time: {end_time}')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
