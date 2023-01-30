import glob

from absl import app, logging, flags
from absl.flags import FLAGS
import tensorflow as tf
import tqdm

flags.DEFINE_enum('orientation', 'axial', ['axial', 'coronal', 'sagittal'], 
                  'axial: axial orientation, ' 
                  'coronal: coronal orientation, ' 
                  'sagittal: sagittal orientation')

def build_example(img_path, label, xmax, ymax, xmin, ymin, class_list):

    img_raw = open(img_path, 'rb').read()
    classes_text = list(map(lambda x: class_list[x].encode('utf8'), label))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }))
    return example

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

def main(_argv):

    class_list = ["negative", "positive"]

    orientation = FLAGS.orientation

    train_writer = tf.io.TFRecordWriter(f"./tfrecord-data/{orientation}_brain_train.tfrecord")
    val_writer = tf.io.TFRecordWriter(f"./tfrecord-data/{orientation}_brain_val.tfrecord")

    train_image_list = glob.glob(f"./brain-tumor-data/{orientation}_t1wce_2_class/images/train/*.jpg")
    train_label_list = glob.glob(f"./brain-tumor-data/{orientation}_t1wce_2_class/labels/train/*.txt")

    val_image_list = glob.glob(f"./brain-tumor-data/{orientation}_t1wce_2_class/images/test/*.jpg")
    val_label_list = glob.glob(f"./brain-tumor-data/{orientation}_t1wce_2_class/labels/test/*.txt")

    train_image_set = set(map(lambda x: x.split('/')[-1][:-4], train_image_list))
    train_label_set = set(map(lambda x: x.split('/')[-1][:-4], train_label_list))
    train_id_list = list(train_image_set & train_label_set) # exclude mismatching files

    val_image_set = set(map(lambda x: x.split('/')[-1][:-4], val_image_list))
    val_label_set = set(map(lambda x: x.split('/')[-1][:-4], val_label_list))
    val_id_list = list(val_image_set & val_label_set) # exclude mismatching files

    logging.info("Train image list loaded: %d", len(train_id_list))
    logging.info("Validaiton image list loaded: %d", len(val_id_list))

    for id_list, type in zip([train_id_list, val_id_list], ["train", "test"]):
        writer = train_writer if type == "train" else val_writer
        for id in tqdm.tqdm(id_list):
            label_filename = f"./brain-tumor-data/{orientation}_t1wce_2_class/labels/{type}/{id}.txt"
            image_filename = f"./brain-tumor-data/{orientation}_t1wce_2_class/images/{type}/{id}.jpg"
            with open(label_filename, "r") as f:
                labels, xmaxs, ymaxs, xmins, ymins = [], [], [], [], []
                for line in f.readlines():
                    label, x, y, w, h = line.split(' ')
                    xmin, ymin, xmax, ymax = yolobbox2bbox(float(x), float(y), float(w), float(h))
                    labels.append(int(label))
                    xmaxs.append(xmax)
                    ymaxs.append(ymax)
                    xmins.append(xmin)
                    ymins.append(ymin)
                tf_example = build_example(image_filename, labels, xmaxs, ymaxs, xmins, ymins, class_list)
                writer.write(tf_example.SerializeToString())
        writer.close()
        logging.info(f"{type} data creaetion done")

if __name__ == '__main__':
    app.run(main)
