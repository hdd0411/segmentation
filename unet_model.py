# -*- coding: utf-8 -*-

import  tensorflow as tf  
import  os
import numpy as np

def deconv_2d(x,w,stride):
    """
    反卷积
    """
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def crop_and_concat(x1,x2):
    """
    将x1裁剪成x2尺寸，然后将两个tensor拼接成一个
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3) 
 
def crop_to_shape(data, shape):
    """
    将NWH格式的数据data的W与H裁剪成shape大小
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def selu(x):
    """
    自归一化激活函数
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.nn.elu(x))
    
def pixel_wise_softmax(output_map):
    """
    逐像素计算属于各类别概率
    """
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)

class network(object):  
    def __init__(self, channels, n_class):
        """
        定义U-Net网络，声明网络变量参数
        """
        self.channels = channels
        self.n_classes = n_class 
         
        with tf.variable_scope("weights"):  
            self.weights={  
                #39*39*3->36*36*20->18*18*20  
                'conv11':tf.get_variable('conv11',[3,3,self.channels,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #18*18*20->16*16*40->8*8*40  
                'conv12':tf.get_variable('conv12',[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                 
                #39*39*3->36*36*20->18*18*20  
                'conv21':tf.get_variable('conv21',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #18*18*20->16*16*40->8*8*40  
                'conv22':tf.get_variable('conv22',[3,3,128,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                #39*39*3->36*36*20->18*18*20  
                'conv31':tf.get_variable('conv31',[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #18*18*20->16*16*40->8*8*40  
                'conv32':tf.get_variable('conv32',[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                #39*39*3->36*36*20->18*18*20  
                'conv41':tf.get_variable('conv41',[3,3,256,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #18*18*20->16*16*40->8*8*40  
                'conv42':tf.get_variable('conv42',[3,3,512,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv51':tf.get_variable('conv51',[3,3,512,1024],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv52':tf.get_variable('conv52',[3,3,1024,1024,],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'deconv1':tf.get_variable('deconv1',[3,3,512,1024],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv61':tf.get_variable('conv61',[3,3,1024,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv62':tf.get_variable('conv62',[3,3,512,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'deconv2':tf.get_variable('deconv2',[3,3,256,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv71':tf.get_variable('conv71',[3,3,512,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv72':tf.get_variable('conv72',[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'deconv3':tf.get_variable('deconv3',[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv81':tf.get_variable('conv81',[3,3,256,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv82':tf.get_variable('conv82',[3,3,128,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                 
                'deconv4':tf.get_variable('deconv4',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 
                
                'conv91':tf.get_variable('conv91',[3,3,128,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv92':tf.get_variable('conv92',[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
              
                'conv10':tf.get_variable('conv10',[3,3,64,self.n_classes],initializer=tf.contrib.layers.xavier_initializer_conv2d())  
            }
              
        with tf.variable_scope("biases"):  
            self.biases={                   
                 #39*39*3->36*36*20->18*18*20  
                'conv11':tf.get_variable('conv11',[64],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),  
                #18*18*20->16*16*40->8*8*40  
                'conv12':tf.get_variable('conv12',[64],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                 
                #39*39*3->36*36*20->18*18*20  
                'conv21':tf.get_variable('conv21',[128],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),  
                #18*18*20->16*16*40->8*8*40  
                'conv22':tf.get_variable('conv22',[128],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                #39*39*3->36*36*20->18*18*20  
                'conv31':tf.get_variable('conv31',[256],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),  
                #18*18*20->16*16*40->8*8*40  
                'conv32':tf.get_variable('conv32',[256],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                #39*39*3->36*36*20->18*18*20  
                'conv41':tf.get_variable('conv41',[512],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),  
                #18*18*20->16*16*40->8*8*40  
                'conv42':tf.get_variable('conv42',[512],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'conv51':tf.get_variable('conv51',[1024],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                'conv52':tf.get_variable('conv52',[1024,],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'deconv1':tf.get_variable('deconv1',[512],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'conv61':tf.get_variable('conv61',[512],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                'conv62':tf.get_variable('conv62',[512],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'deconv2':tf.get_variable('deconv2',[256],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'conv71':tf.get_variable('conv71',[256],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                'conv72':tf.get_variable('conv72',[256],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'deconv3':tf.get_variable('deconv3',[128],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                
                'conv81':tf.get_variable('conv81',[128],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                'conv82':tf.get_variable('conv82',[128],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                 
                'deconv4':tf.get_variable('deconv4',[64],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)), 
                
                'conv91':tf.get_variable('conv91',[64],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
                'conv92':tf.get_variable('conv92',[64],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),
              
                'conv10':tf.get_variable('conv10',[self.n_classes],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32)),  
            }
            
    def inference(self, inimg, keep_prob):
        """
        网络前向传播计算，输出logits张量，keep_prob为drop out参数，预测时置为1
        """
        nx = tf.shape(inimg)[1];
        ny = tf.shape(inimg)[2];
    
        in_node = tf.reshape(inimg,tf.stack([-1,nx,ny,self.channels]));
        
        'U-Shape 左端网络结构，feature map 尺寸变小'
        conv2d_11 = tf.nn.conv2d(in_node, self.weights['conv11'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d_11+self.biases['conv11']), keep_prob)
        conv2d_12 = tf.nn.conv2d(tmp_h_conv1, self.weights['conv12'], strides=[1,1,1,1],padding='SAME')
        conv1 = tf.nn.dropout(tf.nn.relu(conv2d_12+self.biases['conv12']), keep_prob)
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        
        conv2d_21 = tf.nn.conv2d(pool1, self.weights['conv21'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d_21+self.biases['conv21']), keep_prob)
        conv2d_22 = tf.nn.conv2d(tmp_h_conv2, self.weights['conv22'], strides=[1,1,1,1],padding='SAME')
        conv2 = tf.nn.dropout(tf.nn.relu(conv2d_22+self.biases['conv22']), keep_prob)
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        
        conv2d_31 = tf.nn.conv2d(pool2, self.weights['conv31'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv3 = tf.nn.dropout(tf.nn.relu(conv2d_31+self.biases['conv31']), keep_prob)
        conv2d_32 = tf.nn.conv2d(tmp_h_conv3, self.weights['conv32'], strides=[1,1,1,1],padding='SAME')
        conv3 = tf.nn.dropout(tf.nn.relu(conv2d_32+self.biases['conv32']), keep_prob)
        pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')  
        
        conv2d_41 = tf.nn.conv2d(pool3, self.weights['conv41'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv4 = tf.nn.dropout(tf.nn.relu(conv2d_41+self.biases['conv41']), keep_prob)
        conv2d_42 = tf.nn.conv2d(tmp_h_conv4, self.weights['conv42'], strides=[1,1,1,1],padding='SAME')
        conv4 = tf.nn.dropout(tf.nn.relu(conv2d_42+self.biases['conv42']), keep_prob)
        pool4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') 
        
        conv2d_51 = tf.nn.conv2d(pool4, self.weights['conv51'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv5 = tf.nn.dropout(tf.nn.relu(conv2d_51+self.biases['conv51']), keep_prob)
        conv2d_52 = tf.nn.conv2d(tmp_h_conv5, self.weights['conv52'], strides=[1,1,1,1],padding='SAME')
        conv5 = tf.nn.dropout(tf.nn.relu(conv2d_52+self.biases['conv52']), keep_prob)
        
        
        'U-Shape 右端网络结构，上卷积开始，feature map 尺寸变大'
        h_deconv1 = tf.nn.relu(deconv_2d(conv5, self.weights['deconv1'], 2)+self.biases['deconv1'])
        h_deconv_concat_1 = crop_and_concat(conv4,h_deconv1)
        conv2d_61 = tf.nn.conv2d(h_deconv_concat_1, self.weights['conv61'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv6 = tf.nn.dropout(tf.nn.relu(conv2d_61+self.biases['conv61']), keep_prob)
        conv2d_62 = tf.nn.conv2d(tmp_h_conv6, self.weights['conv62'], strides=[1,1,1,1],padding='SAME')
        conv6 = tf.nn.dropout(tf.nn.relu(conv2d_62+self.biases['conv62']), keep_prob)             
        
        h_deconv2 = tf.nn.relu(deconv_2d(conv6, self.weights['deconv2'], 2)+self.biases['deconv2'])
        h_deconv_concat_2 = crop_and_concat(conv3,h_deconv2)
        conv2d_71 = tf.nn.conv2d(h_deconv_concat_2, self.weights['conv71'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv7 = tf.nn.dropout(tf.nn.relu(conv2d_71+self.biases['conv71']), keep_prob)
        conv2d_72 = tf.nn.conv2d(tmp_h_conv7, self.weights['conv72'], strides=[1,1,1,1],padding='SAME')
        conv7 = tf.nn.dropout(tf.nn.relu(conv2d_72+self.biases['conv72']), keep_prob)
        
        h_deconv3 = tf.nn.relu(deconv_2d(conv7, self.weights['deconv3'], 2)+self.biases['deconv3'])
        h_deconv_concat_3 = crop_and_concat(conv2,h_deconv3)
        conv2d_81 = tf.nn.conv2d(h_deconv_concat_3, self.weights['conv81'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv8 = tf.nn.dropout(tf.nn.relu(conv2d_81+self.biases['conv81']), keep_prob)
        conv2d_82 = tf.nn.conv2d(tmp_h_conv8, self.weights['conv82'], strides=[1,1,1,1],padding='SAME')
        conv8 = tf.nn.dropout(tf.nn.relu(conv2d_82+self.biases['conv82']), keep_prob)
        
        h_deconv4 = tf.nn.relu(deconv_2d(conv8, self.weights['deconv4'], 2)+self.biases['deconv4'])
        h_deconv_concat_4 = crop_and_concat(conv1,h_deconv4)
        conv2d_91 = tf.nn.conv2d(h_deconv_concat_4, self.weights['conv91'], strides=[1,1,1,1],padding='SAME') 
        tmp_h_conv9 = tf.nn.dropout(tf.nn.relu(conv2d_91+self.biases['conv91']), keep_prob)
        conv2d_92 = tf.nn.conv2d(tmp_h_conv9, self.weights['conv92'], strides=[1,1,1,1],padding='SAME')
        conv9 = tf.nn.dropout(tf.nn.relu(conv2d_92+self.biases['conv92']), keep_prob)   
        
        conv2d_10 = tf.nn.conv2d(conv9, self.weights['conv10'], strides=[1,1,1,1],padding='SAME')      
        output_map = tf.nn.relu(conv2d_10+self.biases['conv10'])
        
        
        return output_map
    
    def get_cost(self, logits, labels, cost_name = "cross_entropy", cost_kwargs = {}):
        """
        计算loss,使用交叉熵损失，参数各标签类别权重class_weights及L2正则化系数    
        """
        
        flat_logits = tf.reshape(logits, [-1, self.n_classes])
        flat_labels = tf.reshape(labels, [-1, self.n_classes])
        class_weights = cost_kwargs.pop("class_weights", None)
            
        if class_weights is not None:
            '带权重标签，原文中为了让网络学习区分相邻细胞边界，对距离相邻细胞边界越近的点给予更大的权重'
            class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)
        
            loss_map = tf.nn.softmax_cross_entropy_with_logits(flat_logits, flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
        
            self.loss = tf.reduce_mean(weighted_loss)
                
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))
        
        'L2正则化'
        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            l2_loss = tf.add_n([tf.nn.l2_loss(self.weights[w]) for w in self.weights])
            '一般不对偏置进行正则化，进行正则化算法性能差别也不大'
            #l2_loss += tf.add_n([tf.nn.l2_loss(self.biases[b]) for b in self.biases])
        
            self.loss +=  (regularizer * l2_loss)
            
        return self.loss
    
    def get_accuracy(self, logits, labels):
        """
        计算准确率，以正确预测分类标签比率衡量
        """
        preditc = pixel_wise_softmax(logits)
        labels = tf.cast(labels,tf.float32)
        correct_pred = tf.equal(tf.argmax(preditc, 3), tf.argmax(labels, 3))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return self.accuracy
    
    def get_optimizer(self, loss, training_iters,optimizer = "momentum", opt_kwargs={}):
        """
        设置优化方法及参数
        """
        global_step = tf.Variable(0)
        if optimizer == "momentum":
            learning_rate = opt_kwargs.pop("learning_rate", 0.1)
            decay_rate = opt_kwargs.pop("decay_rate", 0.95)
            momentum = opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            self.opti = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **opt_kwargs).minimize(loss, 
                                                                                global_step=global_step)
        elif optimizer == "adam":
            learning_rate = opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            self.opti = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **opt_kwargs).minimize(loss,
                                                                     global_step=global_step)
        elif optimizer == "sgd":
            learning_rate = opt_kwargs.pop("learning_rate", 0.1)
            decay_rate = opt_kwargs.pop("decay_rate", 0.95)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate)
            
            self.opti = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_node).minimize(loss,
                                                                     global_step=global_step)
        
        return self.opti
    
    def get_summary(self):
        """
        设置记录tensorboard张量
        """
        tf.summary.scalar("cost", self.loss)
        tf.summary.scalar("accuracy",self.accuracy)
        tf.summary.scalar("learn rate",self.learning_rate_node)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op
