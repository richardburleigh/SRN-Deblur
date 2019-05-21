## Prerequisites
- Python2.7
- Scipy
- Scikit-image
- numpy
- Tensorflow 1.4 with NVIDIA GPU or CPU (cpu testing is very slow)
- Opencv-python

## SETUP env for ubuntu 18.04
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

## Installation
Clone this project to your machine.

```bash
git clone https://github.com/relipa/SRN-Deblur.git
cd SRN-Deblur
```

## Download pretrained models

Run `download_model.sh` inside `checkpoints/` by command:
```bash
sh download_model.sh
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

## Training

We trained our model using the dataset from
[DeepDeblur_release](https://github.com/SeungjunNah/DeepDeblur_release).
Please put the dataset into `training_set/`. And the provided `datalist_gopro.txt`
can be used to train the model.

Extra data: https://competitions.codalab.org/competitions/21475#participate

Hyper parameters such as batch size, learning rate, epoch number can be tuned through command line:

```bash
python run_model.py --phase=train --batch=16 --lr=1e-4 --epoch=4000
```

# Training continuous

Set `--incremental_training` is 1 to training continuous
`--shuffle=0` to not shuffle
`--datalist=mydatalist_shuffle.txt` to use datalist shuffle
`--step=5358000` to training continuous from step 5358000

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

### How to choose

If you would like to compare performance against our method, you can use
model `gray` and `lstm`.
If you want to restore blurry images you can try `gray` and `color`.
And `color` is very useful in low-light noisy images.

## Test Video
`--video_filepath_input=./blur.mp4` to set video input
`--video_filepath_output=./result.mp4` to set video output
You need remove all image of `input_path` and `output_path` to store list frames of video

```bash
rm -f testing_set/* && rm -f testing_res/* && python run_model.py --gpu=0 --phase=testVideo --model=color --video_filepath_input=./blur.mp4
```


### Evaluation
`--type`: determine whether video or image
`--gpu`: use gpu or cpu
`--input_path_1`: input path 1 for compare images
`--input_path_2`: input path 2 for compare images
`--video_input_1`: fill path file input 1 for compare video
`--video_input_2`: fill path file input 2 for compare video
`--max_val`: max bit in images (default: 255.0)

```bash
python evaluation.py --video_input_1=./blur.mp4 --video_input_2=./origin.mp4 --type=video
```
