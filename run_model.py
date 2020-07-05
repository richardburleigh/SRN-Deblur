import os
import argparse
import tensorflow as tf
# import models.model_gray as model
# import models.model_color as model
import models.model as model


def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='test', help='determine whether train or test or testVideo')
    parser.add_argument('--datalist', type=str, default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--incremental_training', type=int, default=0, help='continue training with saved model or not')
    parser.add_argument('--shuffle', type=int, default=1, help='shuffle datalist and save')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default='./testing_set',
                        help='input path for testing images')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='output path for testing images')
    parser.add_argument('--video_filepath_input', type=str, default='./test.mp4',
                        help='fill path file input for test video')
    parser.add_argument('--video_filepath_output', type=str, default='./result.mp4',
                        help='fill path file output for test video')
    parser.add_argument('--video_filepath_origin', type=str, default='./origin.mp4',
                        help='fill path file origin for test video')
    parser.add_argument('--origin_path', type=str, default='./origin_img',
                        help='input path for origin images')
    parser.add_argument('--show_evaluation', type=int, default=0,
                        help='flag show evaluation')
    parser.add_argument('--step', type=int, default=None,
                        help='input step to use model')


    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # set up deblur models
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.test(args.height, args.width, args.input_path, args.output_path, args.step)
    elif args.phase == 'train':
        deblur.train()
    elif args.phase == 'testVideo':
        deblur.testVideo(args.height, args.width, args.input_path, args.output_path, args.step, args.video_filepath_input, args.video_filepath_output)
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()