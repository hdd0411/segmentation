#coding=utf-8  
from matplotlib import pyplot as plt
import  tensorflow as tf  
import data_encoder_decoder as indata
import unet_model as model
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import numpy as np
import cv2
import argparse






 

...


IMAGE_SIZE = 512
IMAGE_CHANNEL = 1
NUM_CLASS = 2
BATCH_SIZE = 2
TRAIN_SIZE = 2000
VALIDATE_SIZE = 29

drop_rate = 0.7
l2_weight = 0.0001
optimizer = 'adam'
learning_rate = 0.0002
train_epochs = 50
    
def train(iters, epochs,train_path, validate_path, output_path): 
    """
    训练U-Net网络，iters为迭代步数=样本数/BATCH_SIZE
    train_path为训练集路径，validate_path为验证集路径，output_path为输出目录
    """
    image,label=indata.decode_from_tfrecords(train_path)
    print("***:",image.shape)
    batch_image,batch_label = indata.get_batch(image,label,batch_size=BATCH_SIZE)  
    net = model.network(IMAGE_CHANNEL, NUM_CLASS)

    inf  = net.inference(batch_image,keep_prob=drop_rate)
    loss =  net.get_cost(logits = inf, labels = batch_label,cost_kwargs={'regularizer':l2_weight}) #
    opti = net.get_optimizer(loss = loss , training_iters = iters, optimizer =optimizer,opt_kwargs={'learning_rate':0.0002})
 
    test_image,test_label=indata.decode_from_tfrecords(validate_path)
    test_imgs,test_labs = indata.get_test_batch(test_image,test_label,batch_size=1)  
    test_inf = net.inference(test_imgs,keep_prob=1)
    accuracy = net.get_accuracy(logits = test_inf, labels =test_labs)
    
    summary_op = net.get_summary()

    init=tf.global_variables_initializer()  
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95 # 占用GPU40%的显存
    with tf.Session(config=config) as session: 
        session.run(init)
        session.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)  
        
        ##summary_writer = tf.summary.FileWriter(output_path, graph=session.graph)
        
        if os.path.exists(os.path.join(output_path,'model.ckpt')) is True:  
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join(output_path,'model.ckpt'))    
        saver=tf.train.Saver(max_to_keep=2)
        for epoch in range(epochs):
            
            print('epochs:', epoch)
            
            #for step in range(int(epoch*iters), int((epoch+1)*iters)):
            for step in range(30):
                step += 1
                print('step:',step)
                loss_np,_,label_np,image_np,inf_np=session.run([loss,opti,batch_label,batch_image,inf])                        

                if step%10==0:
                    print ('training steps:', step, ' trainloss:',loss_np ) 
                if step%20==0:  # 15
                    run_count = 0
                    total_accuracy = 0
                    for run_count in range(VALIDATE_SIZE):
                        accuracy_np=session.run(accuracy)
                        total_accuracy += accuracy_np
                    accuracy_np = total_accuracy / VALIDATE_SIZE
                    summary_acc = tf.Summary(value=[
                        tf.Summary.Value(tag="validate_accuracy", simple_value=accuracy_np), 
                    ])
                    ##summary_writer.add_summary(summary_acc, step)
                    ##summary_writer.flush()
                    print ('********test accruacy:',accuracy_np,' at steps:', step, '********')
                    saver.save(session, os.path.join(output_path,'model.ckpt'))
                    print ('writing checkpoint at steps:' , step)
  
        coord.request_stop() 
        coord.join(threads) 
        #summary_writer.close()
        session.close() 

def load_model(session,checkpoint_dir, exclusions=[]):
    """
    加载模型
    """
    net=model.network(IMAGE_CHANNEL, NUM_CLASS)       
    init=tf.global_variables_initializer() 
    session.run(init)
    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ('Restore the model from checkpoint', ckpt.model_checkpoint_path)
        '恢复模型'
        variables_to_restore = []
        for var in tf.trainable_variables():#slim.get_model_variables()
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
            
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(session, ckpt.model_checkpoint_path)
    
    return net 


def test(model_path, data_root):  
    image_filenames=os.listdir(data_root)  
    image_filenames=[(data_root+'/'+i) for i in image_filenames]  
    x = tf.placeholder("float", shape=[IMAGE_SIZE, IMAGE_SIZE])
    x_1 = tf.reshape(x,[1,IMAGE_SIZE,IMAGE_SIZE,1])  

    init=tf.global_variables_initializer() 
        
    with tf.Session() as session:  
        #net=model.network(IMAGE_CHANNEL, NUM_CLASS)
        #tf.train.Saver(max_to_keep=None).restore(session, os.path.join(output_path,'model.ckpt')) 
        net=load_model(session,model_path)
          
        y = net.inference(x_1,keep_prob=1)
        predict_label=tf.argmax(model.pixel_wise_softmax(y), 3)   
        
        print (x_1.get_shape())
        
        for imgf in image_filenames:  
            image=tif.imread(imgf)   
            print ("cv shape:",image.shape) 
  
            y_np=session.run(predict_label,feed_dict = {x:image})
            y_np=y_np.reshape(IMAGE_SIZE,IMAGE_SIZE)    
            print ("cv shape:",y_np.shape)
            plt.figure()
            plt.imshow(y_np, cmap ='gray')
            plt.show()
            plt.figure()
            plt.imshow(image, cmap ='gray')
            plt.show()           
  
        session.close() 
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help="""\
        determine train or test\
        """
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='/home/a/MKQ/pr_test/data/tfrecord/train_set.tfrecords',
        help="""\
        determine path of trian images\
        """
    )

    parser.add_argument(
        '--validate_dir',
        type=str,
        default='/home/a/MKQ/pr_test/data/tfrecord/validation_set.tfrecords',
        help="""\
        determine path of test images\
        """
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help="""\
        determine maximum training epochs\
        """
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/home/a/MKQ/pr_test/data/tfrecord/test_set.tfrecords',
        help="""\
        determine maximum training epochs\
        """
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/a/MKQ/pr_test/data/model/',
        help="""\
        determine maximum training epochs\
        """
    )

    FLAGS = parser.parse_args()

    #FLAGS.mode = 'train'
    if FLAGS.mode == 'train':
        train(TRAIN_SIZE/BATCH_SIZE, FLAGS.max_epochs, FLAGS.train_dir, FLAGS.validate_dir,FLAGS.model_dir)

    elif FLAGS.mode == 'test':
        test(FLAGS.model_dir, FLAGS.test_dir)
    else:
        raise Exception('error mode')
    print('done')
    
