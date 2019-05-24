import os
import argparse
import tensorflow as tf
import cv2
import threading
import scipy.misc
import time
import threading
import multiprocessing
# import models.model_gray as model
# import models.model_color as model

def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--type', type=str, default='video', help='determine whether video or image')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--input_path_1', type=str, default='./input_path_1',
                        help='input path 1 for compare images')
    parser.add_argument('--input_path_2', type=str, default='./input_path_2',
                        help='input path 2 for compare images')
    parser.add_argument('--video_input_1', type=str, default='./test.mp4',
                        help='fill path file input 1 for compare video')
    parser.add_argument('--video_input_2', type=str, default='./result.mp4',
                        help='fill path file input 2 for compare video')
    parser.add_argument('--max_val', type=float, default=255.0,
                        help='max bit in images')

    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # session and thread
    if (args.type == 'video'):
        fps1 = videoToFrames(args.video_input_1, args.input_path_1)
        fps2 = videoToFrames(args.video_input_2, args.input_path_2)
    imgsName = os.listdir(args.input_path_1)

    chunk_size = 50
    pool = multiprocessing.Pool()
    pool_args = [];
    for imgName in imgsName:
        img_path_1 = os.path.join(args.input_path_1, imgName)
        img_path_2 = os.path.join(args.input_path_2, imgName)
        pool_args.append((img_path_1, img_path_2, args.max_val))
        
    results = pool.imap(wrap_compare_img, pool_args, chunksize=chunk_size)
    num = 0
    psnr_total = 0
    ssim_total = 0
    for result in results:
        psnr_total += result[0]
        ssim_total += result[1]
        num +=1
        
    print('psnr_avg: %.4f | ssim_avg: %.4f' % ((psnr_total/num), (ssim_total/num)))

def wrap_compare_img(args):
    return compare_img(*args)

def compare_img(img_path_1, img_path_2, max_val=255.0):
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    img1 = tf.image.decode_jpeg(tf.read_file(img_path_1))
    img2 = tf.image.decode_jpeg(tf.read_file(img_path_2))
    start = time.time()
    psnr = sess.run(tf.image.psnr(img1, img2, max_val))
    ssim = sess.run(tf.image.ssim(img1, img2, max_val))
    duration = time.time() - start
    print('path: %s | psnr: %.4f | ssim: %.4f ... calc_duration: %4.3fs' % (img_path_1, psnr, ssim, duration))
    f = open('./output.txt', 'a')
    f.write('path: %s | psnr: %.4f | ssim: %.4f ... calc_duration: %4.3fs \n' % (img_path_1, psnr, ssim, duration))
    f.close()

    return [psnr, ssim]

def videoToFrames(video_filepath, input_path='./testing_set'):
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

if __name__ == '__main__':
    tf.app.run()
