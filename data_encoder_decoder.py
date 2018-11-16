# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import cv2
    
def  encode_to_tfrecords(lable_file,data_root,new_name='data.tfrecords'):
    """
    将数据转成TFRecords格式，其中lable_file为数据list文件，图像与标签文件名中间以空格隔开
    data_root为保存数据目录，new_name为数据名
    """
      
    writer=tf.python_io.TFRecordWriter(data_root+'/'+new_name)  
    num_example=0  
    with open(lable_file,'r') as f:  
        for l in f.readlines():  
            l = l.strip('\n').split(" ") 
            image=cv2.imread(l[0])
            label=cv2.imread(l[1]) 
  
            example=tf.train.Example(features=tf.train.Features(feature={  
#                 'height':tf.FixedLenFeature([],tf.int64),  
#                 'width':tf.FixedLenFeature([],tf.int64),  
#                 'nchannel':tf.FixedLenFeature([],tf.int64),  
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))  
            }))  
            serialized=example.SerializeToString()  
            writer.write(serialized)  
            num_example+=1  
    print(lable_file,"样本数据量：",num_example)
    writer.close()  
      
def decode_from_tfrecords(filename,image_width = 512, image_height = 512, image_channels = 1, num_epoch=None):
    """
       将TFRecord格式数据解码成图像格式
    """ 
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)

    reader=tf.TFRecordReader()  
    _,serialized=reader.read(filename_queue)  
    example=tf.parse_single_example(
        serialized,
        features={  
            'image_raw':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.string)  
        })
    
#     height = tf.cast(example['height'], tf.int32)
#     width = tf.cast(example['width'], tf.int32)
#     channel = tf.cast(example['nchannel'], tf.int32) 
 
    image=tf.decode_raw(example['image_raw'],tf.uint8)
    label=tf.decode_raw(example['label'],tf.uint8)
    
    image = tf.reshape(image,[image_width,image_height,image_channels])
    label = tf.reshape(label,[image_width,image_height,image_channels])
    '归一化'
    image = tf.cast(image,tf.float32)*1.0/255 - 0.5
    '生存one_hot标签'
    label = tf.cast(label,tf.bool)
    label2 = tf.logical_not(label)
    new_label = tf.concat([label2,label], 2)   
    return image,new_label 
  
def get_batch(image, label,batch_size):
    """
    生成乱序mini batch
    """ 
    
    '进行随机翻转/亮度/对比度调整'
    image_shape = image.shape
    image_and_label = tf.concat([image,tf.cast(label,tf.float32)],2)
    image_and_label = tf.image.random_flip_left_right(image_and_label);
    image = tf.reshape(image_and_label[:,:,0], image_shape);
    label = tf.cast(image_and_label[:,:,1:3],tf.bool);
    image = tf.image.random_brightness(image,max_delta=0.24)#亮度变化  
    image = tf.image.random_contrast(image,lower=0.2, upper=1.8)#对比度变化  
    image_batch, label_batch = tf.train.shuffle_batch([image, label],batch_size=batch_size,num_threads=64,capacity=200,min_after_dequeue=50 )

    return image_batch, label_batch   


def get_test_batch(image, label, batch_size):
    """
    生成mini batch，预测时使用
    """  
    images, label_batch=tf.train.batch([image, label],batch_size=batch_size)  
    return images, label_batch   
  
  
def testread(path):
    """
    测试数据格式转换是否有误
    """  
    image,label=decode_from_tfrecords(path)
    print(path)
    images, sparse_labels=get_batch(image, label, 1)
    
    init=tf.global_variables_initializer()

    with tf.Session() as session:  
        session.run(init)
        session.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=session,coord=coord)
        for l in range(10): 

            images_np,batch_label_np=session.run([images,sparse_labels])  
            print (images_np.shape )
            print (batch_label_np.shape)
            img=np.array(batch_label_np[0,:,:,0], dtype=float)
            img=np.reshape(img, (512,512))
            print (img.shape)
            # plt.figure()
            # plt.imshow(img, cmap ='gray')
            # plt.show()
        
        '关闭queue'             
        coord.request_stop() 
        coord.join(threads)
    
# if __name__=='__main__':
#     testread('/home/a/MKQ/project/data/tfrecord/train_set.tfrecords')
    
