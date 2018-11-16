# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import Augmentor
import cv2
import glob

TRAIN_SET_NAME='train_set.tfrecords'
VALIDATION_SET_NAME='validation_set.tfrecords'
TEST_SET_NAME='test_set.tfrecords'
PREDICT_SET_NAME='predict_set.tfrecords'

ORIGIN_IMAGES_DIRECTORY='./data/images'
ORIGIN_LABEL_DIRECTORY='./data/ground_truth_images'
AUGMENT_IMAGES_PATH='./data/augment_images'
AUGMENT_LABEL_PATH='./data/augment_label'
AUGMENT_DATA_PATH= \
    os.getcwd()[:os.getcwd().rindex('/')]+'/data/augment_data'
IMAGE_SET='./data/image_set'
LABEL_SET='./data/label_set'



INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_CHANNEL=512,512,1
OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT,OUTPUT_IMAGE_CHANNEL=512,512,1

TRAIN_SET_SIZE=1900
VALIDATION_SET_SIZE=230
TEST_SET_SIZE=27
PREDICT_SET_SIZE=30

def augment():
    p=Augmentor.Pipeline(ORIGIN_IMAGES_DIRECTORY)
    p.ground_truth(ORIGIN_LABEL_DIRECTORY)
    p.rotate(probability=0.2,max_left_rotation=2,max_right_rotation=2)
    p.zoom(probability=0.2,min_factor=1.1,max_factor=1.2)
    p.skew(probability=0.2)
    p.random_distortion(probability=0.2,grid_width=100,grid_height=100,magnitude=1)
    p.shear(probability=0.2,max_shear_left=2,max_shear_right=2)
    p.crop_random(probability=0.2,percentage_area=0.8)
    p.flip_random(probability=0.2)
    p.sample(n=TRAIN_SET_SIZE+VALIDATION_SET_SIZE+TEST_SET_SIZE)

def rename():
    augment_images_path=glob.glob(os.path.join(AUGMENT_IMAGES_PATH,'*.bmp'))  ## 获得指定路径下面的bmp文件路径
    augment_labels_path=glob.glob(os.path.join(AUGMENT_LABEL_PATH,'*.bmp'))
    augment_images_path.sort()
    augment_labels_path.sort()
    images_path=IMAGE_SET
    labels_path=LABEL_SET
    for index,image_path in enumerate(augment_images_path):
        image=cv2.imread(image_path)
        cv2.imwrite(filename=os.path.join(images_path, '%d.bmp' % index),img=image)
    for index,label_path in enumerate(augment_labels_path):
        label=cv2.imread(label_path)
        cv2.imwrite(filename=os.path.join(labels_path, '%d.bmp' % index),img=label)           
    print('Done rename!')
        
def image_to_tfrecord():
    images_path=IMAGE_SET
    labels_path=LABEL_SET
    train_set_writer=tf.python_io.TFRecordWriter(os.path.join('./data/tfrecord',TRAIN_SET_NAME))
    validation_set_writer=tf.python_io.TFRecordWriter(os.path.join('./data/tfrecord',VALIDATION_SET_NAME))
    test_set_writer=tf.python_io.TFRecordWriter(os.path.join('./data/tfrecord',TEST_SET_NAME))
    
    #train_set
    for index in range(TRAIN_SET_SIZE):
        train_image=cv2.imread(os.path.join(images_path, '%d.bmp' % index),flags=0)
        train_label=cv2.imread(os.path.join(labels_path, '%d.bmp' % index),flags=0)
        train_image=cv2.resize(src=train_image,dsize=(INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT))
        train_label=cv2.resize(src=train_label,dsize=(INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT))
        train_label[train_label<=100]=0
        train_label[train_label>100]=1
        example=tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_label.tobytes()])), ### transform array to bytes
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image.tobytes()]))
        }))
        train_set_writer.write(example.SerializeToString())
        if index % 100 == 0:
            print('Done train_set writing %.2f%%' % (index / TRAIN_SET_SIZE * 100))
        train_set_writer.close()
        print('Done train_set writing!')

    #validation_set
    for index in range(TRAIN_SET_SIZE,TRAIN_SET_SIZE+VALIDATION_SET_SIZE):
        validation_image=cv2.imread(os.path.join(images_path, '%d.bmp' % index),flags=0)
        validation_label=cv2.imread(os.path.join(images_path, '%d.bmp' % index),flags=0)
        validation_image=cv2.resize(src=validation_image,dsize=(INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT))
        validation_label=cv2.resize(src=validation_label,dsize=(INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT))
        validation_label[validation_label<=100]=0
        validation_label[validation_label>100]=1
        example=tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_label.tobytes()])),
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_image.tobytes()]))
        }))
        validation_set_writer.write(example.SerializeToString())
        if index%10==0:
            print('Done validation_set writing %.2f%%' % ((index-TRAIN_SET_SIZE)/VALIDATION_SET_SIZE*100))
        validation_set_writer.close()
        print('Done validation_set writing!')
    #test_set
    for index in range(TRAIN_SET_SIZE+VALIDATION_SET_SIZE,TRAIN_SET_SIZE+VALIDATION_SET_SIZE+TEST_SET_SIZE):
        test_image=cv2.imread(os.path.join(images_path, '%d.bmp' % index),flags=0)
        test_label=cv2.imread(os.path.join(images_path, '%d.bmp' % index),flags=0)
        test_image=cv2.resize(src=test_image,dsize=(INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT))
        test_label=cv2.resize(src=test_label,dsize=(INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT))
        test_label[test_label<=100]=0
        test_label[test_label>100]=1
        example=tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_label.tobytes()])),
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_image.tobytes()]))
        }))
        test_set_writer.write(example.SerializeToString())
        if index%10==0:
            print('Done validation_set writing %.2f%%' % ((index-TRAIN_SET_SIZE-VALIDATION_SET_SIZE)/TEST_SET_SIZE*100))
    test_set_writer.close()
    print('Done test_set writing!')


if __name__ == '__main__':
    augment()
    #rename()
    #image_to_tfrecord()
