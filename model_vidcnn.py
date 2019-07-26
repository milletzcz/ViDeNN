import time
import numpy as np
import tensorflow as tf
import cv2

def Spatial_CNN(input, is_training=False,trainable=True,reuse=False):
    '''SPATIAL DENOISING CNN'''
    with tf.variable_scope('Spatial_CNN',reuse=reuse):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(input, 128, 3, padding='same',name='conv1d', activation=tf.nn.relu,trainable=trainable,reuse=reuse)
            print(output.name)
        for layers in range(2, 19+1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False,trainable=trainable,reuse=reuse)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))   
        with tf.variable_scope('block20'):
            output = tf.layers.conv2d(output, 4, 3, padding='same',name='conv20d',use_bias=False,trainable=trainable,reuse=reuse)
        out = tf.subtract(input,output,name='sub')
    return out#input - output




def Temp3_CNN(input):
    '''TEMPORAL DENOISING CNN'''
    choice = 0
    with tf.variable_scope('Temp3_CNN',reuse=False):
        input_middle = input[:,:,:,choice*4:(choice+1)*4]
        with tf.variable_scope('temp-block1'):
            output = tf.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.leaky_relu)
        for layers in range(2, 19+1):
            with tf.variable_scope('temp-block%d' % layers):
                output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False, activation=tf.nn.leaky_relu)
        with tf.variable_scope('temp-block20'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')
        out = tf.subtract(input_middle,output,name='sub')
    return out

def Spatial_Tmp(input,is_training=False):
    frm = 4#tf.shape(input)[-1]
    print(input.shape)
    input_f = input[:,:,:,0:4]
    print(input_f.shape)
    inframes = Spatial_CNN(input_f,is_training=False,trainable=False,reuse=False)#False)#True)
    for f in range(1,frm):
        input_f = input[:,:,:,4*f:4*(f+1)]
        print(input_f.shape)
        inframes_tmp = Spatial_CNN(input_f,is_training=False, trainable=False,reuse=True) 
        inframes = tf.concat([inframes,inframes_tmp], axis=-1)
    print(inframes.shape)
    out = Temp3_CNN(inframes)
    return out

        
class vidcnn(object):
    def __init__(self, sess):
        self.sess = sess
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, 3],name='clean_image')
        self.X = tf.placeholder(tf.float32, [None, None, None, 3],name='noisy_image')
        self.Y = Spatial_CNN(self.X)
        self.Y_frames = tf.placeholder(tf.float32, [None, None, None, 9],name='clean_frames')
        self.Xframes = tf.placeholder(tf.float32, [None, None, None, 9],name='noisy_frames')
        self.Yframes = Temp3_CNN(self.Xframes)        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")


    def test(self, noisy_data, orig_data, ckpt_dir, save_dir):
        """Test VidCNN"""
        # init variables
        tf.global_variables_initializer().run()                   
        assert len(noisy_data) != 0, '[!] No test data in the specified folder! Check that contains an original and noisy folder.'
        load_model_status = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED! Check the checkpoint folder.'       

        psnr_sum = 0
        start = time.time()
        for idx in xrange(len(noisy_data)-1):   
            if idx==0:
                test = cv2.imread(orig_data[idx])
                test1 = cv2.imread(orig_data[idx+1])
                test2 = cv2.imread(orig_data[idx+2])
                noisy = cv2.imread(noisy_data[idx])
                noisy1 = cv2.imread(noisy_data[idx+1])
                noisy2 = cv2.imread(noisy_data[idx+2])
                
                test = test.astype(np.float32) / 255.0
                test1 = test1.astype(np.float32) / 255.0
                test2 = test2.astype(np.float32) / 255.0
                noisy = noisy.astype(np.float32) / 255.0
                noisy1 = noisy1.astype(np.float32) / 255.0
                noisy2 = noisy2.astype(np.float32) / 255.0
                
                noisyin2 = np.zeros((1,test.shape[0],test.shape[1],9))    
                current = np.zeros((test.shape[0],test.shape[1],3)) 
                previous = np.zeros((test.shape[0],test.shape[1],3)) 
                
                noisyin = np.zeros((3,test.shape[0],test.shape[1],3))
                noisyin[0] = noisy
                noisyin[1] = noisy1
                noisyin[2] = noisy2 
                out = self.sess.run([self.Y],feed_dict={self.X:noisyin})
                out = np.asarray(out)

                noisyin2[0,:,:,0:3] = out[0,0]
                noisyin2[0,:,:,3:6] = out[0,0]
                noisyin2[0,:,:,6:] = out[0,1]
                temp_clean_image= self.sess.run([self.Yframes],feed_dict={self.Xframes:noisyin2})
                temp_clean_image = np.asarray(temp_clean_image)
                cv2.imwrite(save_dir + '/%04d.png'%idx,temp_clean_image[0,0]*255)
                psnr = psnr_scaled(test,temp_clean_image[0,0])
                psnr_sum += psnr
                print("Frame %d PSNR: %f" % (idx, psnr))

                noisyin2[0,:,:,0:3] = out[0,0]
                noisyin2[0,:,:,3:6] = out[0,1]
                noisyin2[0,:,:,6:] = out[0,2]
                current[:,:,:] = out[0,2,:,:,:]
                previous[:,:,:] = out[0,1,:,:,:]
            else:
                if idx<(len(noisy_data)-2):
                    test3 = cv2.imread(orig_data[idx+2])
                    test3 = test3.astype(np.float32) / 255.0
                    noisy3 = cv2.imread(noisy_data[idx+2])
                    noisy3 = noisy3.astype(np.float32) / 255.0
    
                    out2 = self.sess.run([self.Y],feed_dict={self.X:np.expand_dims(noisy3,0)})
                    out2 = np.asarray(out2)
                    
                    noisyin2[0,:,:,0:3] = previous
                    noisyin2[0,:,:,3:6] = current
                    noisyin2[0,:,:,6:] = out2[0,0]
                    previous = current
                    current = out2[0,0]
                else:
                    try:
                        out2
                    except NameError:
                        out2 = np.zeros((out.shape))
                        out2=out
                        out2[0,0]=out[0,2]
                    noisyin2[0,:,:,0:3] = current
                    noisyin2[0,:,:,3:6] = out2[0,0]
                    noisyin2[0,:,:,6:] = out2[0,0]
            temp_clean_image= self.sess.run(
                [self.Yframes],feed_dict={self.Xframes:noisyin2})            
            
            temp_clean_image = np.asarray(temp_clean_image)
            cv2.imwrite(save_dir + '/%04d.png'%(idx+1),temp_clean_image[0,0]*255)

            # calculate PSNR
            if idx==0:
                psnr = psnr_scaled(test1, temp_clean_image[0,0])
            else:
                psnr = psnr_scaled(test2, temp_clean_image[0,0])
                try: #need this when testing with only 3 frames
                    test3
                except NameError:
                    test3=test2
                test2=test3
                
            print("Frame %d PSNR: %f" % (idx+1, psnr))
            psnr_sum += psnr
        avg_psnr = psnr_sum / len(orig_data)
        print("### Average PSNR %.2f" % avg_psnr)
        print("### Elapsed time: %.4f" %(time.time()-start))


    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(self.sess, full_path)
            return True
        else:
            return False, 0

    
def psnr_scaled(im1, im2): # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 **2 / mse)
    return psnr
