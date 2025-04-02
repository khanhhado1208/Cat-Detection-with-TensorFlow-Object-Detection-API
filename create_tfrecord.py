import os
import glob
import tensorflow as tf
import xml.etree.ElementTree as ET
import argparse
from object_detection.utils import dataset_util

# Load label map
def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, "r") as f:
        for line in f.readlines():
            if "id:" in line:
                label_id = int(line.split(":")[-1].strip())
            if "name:" in line:
                label_name = line.split(":")[-1].strip().replace('"', '')
                label_map[label_name] = label_id
    return label_map

# Convert annotation from XML → TFRecord
def xml_to_tf_example(xml_file, label_map_dict):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    image_format = filename.split('.')[-1]
    full_path = os.path.join(os.path.dirname(xml_file), filename)

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    for member in root.findall('object'):
        class_name = member.find('name').text
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

        bndbox = member.find('bndbox')
        xmins.append(float(bndbox.find('xmin').text) / width)
        xmaxs.append(float(bndbox.find('xmax').text) / width)
        ymins.append(float(bndbox.find('ymin').text) / height)
        ymaxs.append(float(bndbox.find('ymax').text) / height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Function create TFRecord from image folder and annotation XML
def convert_to_tfrecord(data_dir, label_map_path, output_path):
    label_map_dict = load_label_map(label_map_path)
    writer = tf.io.TFRecordWriter(output_path)
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))

    for xml_file in xml_files:
        tf_example = xml_to_tf_example(xml_file, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f" TFRecord file created: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--label_map_path", required=True, help="Path to label_map.pbtxt")
    parser.add_argument("--output_path", required=True, help="Path to save the TFRecord file")

    args = parser.parse_args()
    convert_to_tfrecord(args.data_dir, args.label_map_path, args.output_path)
