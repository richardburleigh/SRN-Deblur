from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *
import cv2
import math
from os.path import isfile, join


class DEBLUR(object):
    def videoToFrames(self, video_filepath, input_path='./testing_set'):
        if not os.path.exists(input_path):
            os.makedirs(input_path)

        vidcap = cv2.VideoCapture(video_filepath)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(input_path + "/frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
        vidcap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        fps = 1000 * vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / vidcap.get(cv2.CAP_PROP_POS_MSEC)
        return fps

    def convert_frames_to_video(self, pathIn, pathOut, fps):
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        # for sorting the file names properly
        files.sort(key=lambda x: int(x[5:-4]))

        for i in range(len(files)):
            filename = pathIn + '/' + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            print(filename)
            # inserting the frames into an image array
            frame_array.append(img)

        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()

    def testVideo(self, height, width, input_path, output_path, input_step=523000, video_filepath_input='./test.mp4',
                  video_filepath_output='./result.mp4'):
        if (self.show_evaluation == 1):
            self.videoToFrames(self.video_filepath_origin, self.origin_path)
        fps = self.videoToFrames(video_filepath_input, input_path)
        self.test(height, width, input_path, output_path, input_step)
        self.convert_frames_to_video(output_path, video_filepath_output, fps)

    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        if args.shuffle == 1:
            random.shuffle(self.data_list)
            f = open(os.path.splitext(args.datalist)[0] + '_shuffle.txt', 'w')
            for item in self.data_list:
                f.write(item + "\n")
            f.close()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.origin_path = args.origin_path
        self.video_filepath_origin = args.video_filepath_origin
        self.show_evaluation = args.show_evaluation

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            img_crop = tf.unstack(
                tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                axis=0)
            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in xrange(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                    # encoder
                    conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                    conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                    conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                    conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4

                    # decoder
                    deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                    deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                    deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                    deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                    deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            return x_unwrap

    def build_model(self):
        img_in, img_gt = self.input_producer(self.batch_size)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        # generator
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        for i in xrange(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        init_global_step = 0
        if self.args.incremental_training == 1:
            if self.args.step is not None:
                init_global_step = self.args.step
            else:
                ckpt = tf.train.get_checkpoint_state(self.train_dir)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                ckpt_iter = ckpt_name.split('-')[1]
                init_global_step = int(ckpt_iter)
        self.max_steps += init_global_step

        global_step = tf.Variable(initial_value=init_global_step, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(self.all_vars, max_to_keep=50, keep_checkpoint_every_n_hours=1)
        if self.args.incremental_training == 1:
            self.load(sess, self.train_dir, step=self.args.step)
        self.sess = sess
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                    0.0, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == self.max_steps:
                # checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, self.train_dir, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path, input_step=523000):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=input_step)

        psnr_total = 0.0
        psnr_num = 0
        ssim_total = 0.0
        ssim_num = 0

        for imgName in imgsName:
            blur = scipy.misc.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(output_path, imgName), res)
            if (self.show_evaluation == 1):
                original = tf.image.decode_jpeg(tf.read_file(os.path.join(self.origin_path, imgName)))
                contrast = tf.image.decode_jpeg(tf.read_file(os.path.join(output_path, imgName)))
                start = time.time()
                psnr = sess.run(tf.image.psnr(original, contrast, max_val=255.0))
                ssim = sess.run(tf.image.ssim(original, contrast, max_val=255.0))
                duration = time.time() - start
                print('psnr: %.4f | ssim: %.4f ... calc_duration: %4.3fs' % (psnr, ssim, duration))
                psnr_total += psnr
                psnr_num += 1
                ssim_total += ssim
                ssim_num += 1

        if (psnr_num > 0):
            print('psnr_avg: %.4f | ssim_avg: %.4f' % ((psnr_total/psnr_num), (ssim_total/ssim_num)))


    def psnr(self, img1, img2, pixel_max = 255.0):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        return 20 * math.log10(pixel_max / math.sqrt(mse))
    def load_image(self, pathfile):
        with open (pathfile, 'rb') as f:
            return np.array(f.read())