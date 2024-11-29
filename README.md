## RTDETR-PyTorch
This repositroy fork by [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR).

It make for provides better pytorch code.

- easier to debug than the original pytorch code.
- easier to read than the original pytorch code.
- Don't use YML config files. You only need to look at the code.
- Check out the model class zoo. [src/zoo/model.py](https://github.com/int11/RT_DETR_Pytorch/blob/main/src/zoo/model.py), [src/zoo/optimizer.py](https://github.com/int11/RT_DETR_Pytorch/blob/main/src/zoo/optimizer.py)
- Check out the training example [tools/train.py](https://github.com/int11/RT_DETR_Pytorch/blob/main/tools/train.py)
- In training, coco dataset uses less memory with memory share.
- Check out my personal research and anyone is welcome to contribute. [branch/research](https://github.com/int11/RTDETR-PyTorch/tree/research)

## Model Zoo

| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS |  checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
rtdetr_r18vd | COCO | 640 | 46.4 | 63.7 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)
rtdetr_r34vd | COCO | 640 | 48.9 | 66.8 | 31 | 161 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth)
rtdetr_r50vd_m | COCO | 640 | 51.3 | 69.5 | 36 | 145 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth)
rtdetr_r50vd | COCO | 640 | 53.1 | 71.2| 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)
rtdetr_r101vd | COCO | 640 | 54.3 | 72.8 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth)
rtdetr_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetr_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetr_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth)

Notes
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
- `url`<sup>`*`</sup> is the url of pretrained weights convert from paddle model for save energy. *It may have slight differences between this table and paper*
<!-- - `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$ and $tensorrt\\_fp16$ mode -->

## Quick start
It is recommended that you work by changing the default value of the parser.add_argument function to your environment.

<details open>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details open>
<summary>Data</summary>

- Download and extract COCO 2017 train and val images. https://cocodataset.org/#download
- The directory must contain train2017, val2017, and annotation folders.
- When training, enter the dataset path through the --dataset_dir flag (default: dataset/coco)"
```
path/to/dataset
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

</details>



<details open>
<summary>Training & Evaluation</summary>

- Training:
```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0;
python tools/train.py --dataset_dir path/to/dataset

# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3;
torchrun --nproc_per_node=4 tools/train.py 

# Load the weight file to continue train
export CUDA_VISIBLE_DEVICES=0,1,2,3;
torchrun --nproc_per_node=4 tools/train.py -w path/to/weight_file.pth
```

- Evaluation on Multiple GPUs:
```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3;
torchrun --nproc_per_node=4 tools/train.py -w path/to/weight_file.pth --val true
```

- Flag list
```shell
'--weight_path', '-w', type=str, default=None, 
help='path to the weight file (default: None)'

'--save_dir', '-s', type=str, default='output/rtdetr_r18vd_6x_coco',
help='path to the weight save directory (default: output/rtdetr_r18vd_6x_coco)'

'--dataset_dir', type=str, default='dataset/coco',
help='path to the dataset directory (default: dataset/coco). This is the directory that must contains the train2017, val2017, annotations folder'

'--batch_size', type=int, default=4,
help='mini-batch size (default: 4), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel'

'--num_workers', type=int, default=0,
help='number of data loading workers (default: 0)'

'--test-only', type=str2bool, default=False,
help='if True, only evaluate the model (default: False)'

'--amp', type=str2bool, default=True,
help='When GPU is available, use Automatic Mixed Precision (default: True)'

'--ema', type=str2bool, default=True,
help='Use Exponential Moving Average (default: True)'

'--epoch', type=int, default=100,
help='When test-only is False, this is the number of epochs to train (default: 100)'

'--model_type', type=str, default='r18vd',
choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd'],
help='choose the model type (default: r18vd)'
```
</details>



<details open>
<summary>Export</summary>

- This part remains the same as the source code of the forked repository.
- need to check code and refactor. Anyone please contribute to the code.

</details>




<details open>
<summary>Train custom data</summary>

- Until now, there is no support from the command line. train by modifying the code yourself.

</details>
