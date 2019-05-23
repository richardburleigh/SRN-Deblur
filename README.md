# Installation

## Prerequisites
- Python2.7
- Scipy
- Scikit-image
- numpy
- Tensorflow 1.4 with NVIDIA GPU or CPU (cpu testing is very slow)
- Opencv-python

## Example for ubuntu 18.04
```bash
wget https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh
sha256sum Anaconda2-2019.03-Linux-x86_64.sh
printf '\n\n\n\nyes\n\n\n' | bash Anaconda2-2019.03-Linux-x86_64.sh
source ~/.bashrc
conda install scipy
conda install scikit-image
conda install Tensorflow
conda install numpy
conda install opencv-python
```

### Unziping source code to folder SRN-Deblur

## Downloading pretrained models

Run `download_model.sh` inside `checkpoints/` by command:
```bash
sh download_model.sh
```

# Testing

## Testing with Video

`--video_filepath_input=<INPUT_VIDEO>`: path for input video - default='./test.mp4'<br/>
`--video_filepath_output=.<OUTPUT_VIDEO>`: path for output video - default='./result.mp4'<br/>
`--input_path`: path for storing list frames of input video - default='./testing_set'<br/>
`--output_path`: path for storing list frames of output video - default='./testing_res'<br/>
You need remove all image of `input_path` and `output_path` to store list frames of video before running test with video

If you have a GPU, please include `--gpu` argument, and add your gpu id to your command.
Otherwise, use `--gpu=-1` for CPU.

```bash
rm -f testing_set/* && rm -f testing_res/* && python run_model.py --gpu=0 --phase=testVideo --model=color --video_filepath_input=./blur.mp4
```

## Testing with image

To test blur images in a folder, just use arguments
`--input_path=<TEST_FOLDER>` and save the outputs to `--output_path=<OUTPUT_FOLDER>`.
For example:

```bash
python run_model.py --input_path=./testing_set --output_path=./testing_res
```

If you have a GPU, please include `--gpu` argument, and add your gpu id to your command.
Otherwise, use `--gpu=-1` for CPU.

```bash
python run_model.py --gpu=0
```

To test the model, pre-defined height and width of tensorflow
placeholder should be assigned.
Our network requires the height and width be multiples of `16`.
When the gpu memory is enough, the height and width could be assigned to
the maximum to accommodate all the images.

Otherwise, the images will be downsampled by the largest scale factor to
be fed into the placeholder. And results will be upsampled to the original size.

According to our experience, `--height=720` and `--width=1280` work well
on a Gefore GTX 1050 TI with 4GB memory. For example,

```bash
python run_model.py --height=720 --width=1280
```

# Training

We trained our model using the dataset from
[DeepDeblur_release](https://github.com/SeungjunNah/DeepDeblur_release).
Please put the dataset into `training_set/`. And the provided `datalist_gopro.txt`
can be used to train the model.

Extra data: https://competitions.codalab.org/competitions/21475#participate

Hyper parameters such as batch size, learning rate, epoch number can be tuned through command line:

```bash
python run_model.py --phase=train --batch=16 --lr=1e-4 --epoch=4000
```

## Continuous Training

Set `--incremental_training` is 1 to training continuous - default=0<br/>
`--shuffle=0` to not shuffle - default=1<br/>
`--datalist=mydatalist_shuffle.txt` to use datalist shuffle - default='./datalist_gopro.txt'<br/>
`--step=5358000` to training continuous from step 5358000 - default=None

```bash
python run_model.py --phase=train --batch=16 --lr=1e-4 --epoch=10 --incremental_training=1 --datalist=mydatalist_shuffle.txt --shuffle=0 --step=5358000
```

## Models

We provided 3 models (training settings) for testing:
1. `--model=lstm`: This model implements exactly the same structure in our paper.
Current released model weights should produce `PSNR=30.19, SSIM=0.9334` on GOPRO testing dataset.
2. `--model=gray`: According to our further experiments after paper acceptance, we are able
to get a slightly better model by tuning parameters, even without LSTM.
This model should produce visually sharper and quantitatively better results.
3. `--model=color`: Previous models are trained on gray images, and may produce color
ringing artifacts. So we train a model directly based on RGB images.
This model keeps better color consistency, but the results are less sharp.

# All Params

`--phase`: determine whether train or test or testVideo - default='test'<br/>
`--datalist`: training datalist - default='./datalist_gopro.txt'<br/>
`--model`: model type: [lstm | gray | color] - default='color'<br/>
`--incremental_training`: continue training with saved model or not - default=0<br/>
`--shuffle`: shuffle datalist and save - default=1<br/>
`--batch_size`: training batch size - default=16<br/>
`--epoch`: training epoch number - default=4000<br/>
`--lr`: initial learning rate - default=1e-4<br/>
`--gpu`: use gpu or cpu - default='0' (=-1 for using cpu)<br/>
`--height`: height for the tensorflow placeholder, should be multiples of 16 - default=720<br/>
`--width`: width for the tensorflow placeholder, should be multiple of 16 for 3 scales - default=1280<br/>
`--input_path`: input path for testing images - default='./testing_set'<br/>
`--output_path`: output path for testing images - default='./testing_res'<br/>
`--video_filepath_input`: input path for testing video - default='./test.mp4'<br/>
`--video_filepath_output`: output path for testing video - default='./result.mp4'<br/>
`--video_filepath_origin`: original file path for evaluating output video - default='./origin.mp4'<br/>
`--origin_path`: original file path for evaluating output image - default='./origin_img'<br/>
`--show_evaluation`: show evaluation after testing - default=0<br/>
`--step`: using model with a specific step - default=None

# Evaluation

`--type`: determine whether video or image - default='video'<br/>
`--gpu`: use gpu or cpu - default='0' (=-1 for using cpu)<br/>
`--input_path_1`: input path 1 for comparing images - default='./input_path_1'<br/>
`--input_path_2`: input path 2 for comparing images - default='./input_path_2'<br/>
`--video_input_1`: input path 1 for comparing video - default='./test.mp4'<br/>
`--video_input_2`: input path 2 for comparing video - default='./result.mp4'<br/>
`--max_val`: maximum possible pixel value of the image - default=255.0

```bash
python evaluation.py --video_input_1=./blur.mp4 --video_input_2=./origin.mp4 --type=video
```
