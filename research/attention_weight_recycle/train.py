import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import zoo
from src.nn.rtdetr.hybrid_encoder import HybridEncoder, TransformerEncoderLayer
from tools.utils import fit, val, str2bool
from src.data.coco.coco_dataset import CocoDetection
from src.misc import dist
from src.data.dataloader import DataLoader
from function import *
from src.nn.rtdetr.rtdetr import RTDETR

def r50vd_encoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1,
        act='silu',
        eval_spatial_size=[640, 640]):
    
    return HybridEncoder2(
        in_channels=in_channels, 
        feat_strides=feat_strides, 
        hidden_dim=hidden_dim,
        use_encoder_idx=use_encoder_idx, 
        num_encoder_layers=num_encoder_layers, 
        nhead=nhead, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout, 
        enc_act=enc_act, 
        pe_temperature=pe_temperature, 
        expansion=expansion, 
        depth_mult=depth_mult, 
        act=act, 
        eval_spatial_size=eval_spatial_size)


def change_forward(model, layer_type, forward_func):
    layers = []
    # 모델의 모든 모듈과 자식 모듈을 순회
    for module in model.modules():
        # 모듈의 타입이 주어진 레이어 타입과 일치하는지 확인
        if isinstance(module, layer_type):
            layers.append(module)

    for layer in layers:
        layer.forward = lambda *args, self=layer, **kwargs: forward_func(self, *args, **kwargs)


def main():
    args = parser.parse_args()

    dist.init_distributed()
    
    # model = getattr(zoo.model, args.model_type)()

    backbone = zoo.model.r50vd_backbone(depth=18, freeze_at=-1, freeze_norm=False)
    encoder = r50vd_encoder(in_channels=[128, 256, 512], expansion=0.5)
    decoder = zoo.model.r50vd_decoder(num_decoder_layers=3)
    
    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )


    change_forward(model, TransformerEncoderLayer, TransformerEncoderLayer_forward)
    optimizer = getattr(zoo.optimizer, args.model_type)(model)

    #TODO There is a slow on a dataset that is not a CocoDetection class, need to fix this
    val_dataset = zoo.coco_val_dataset(
        img_folder=os.path.join(args.dataset_dir, "val2017"),
        ann_file=os.path.join(args.dataset_dir, "annotations/instances_val2017.json"), 
        dataset_class=CocoDetection)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    if args.val:
        val(model=model, 
            weight_path=args.weight_path, 
            val_dataloader=val_dataloader,
            use_amp=args.amp,
            use_ema=args.ema)
    else:
        train_dataset = zoo.coco_train_dataset(
            img_folder=os.path.join(args.dataset_dir, "train2017"),
            ann_file=os.path.join(args.dataset_dir, "annotations/instances_train2017.json"))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, )
        fit(
            model=model, 
            weight_path=args.weight_path,
            optimizer=optimizer, 
            save_dir=args.save_dir, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, 
            use_amp=args.amp, 
            use_ema=args.ema, 
            epoch=args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', '-w', type=str, default=None,
                        help='path to the weight file (default: None)')

    parser.add_argument('--save_dir', '-s', type=str, default='output/attention_weight_recycle',
                        help='path to the weight save directory (default: output/attention_weight_recycle)')

    parser.add_argument('--dataset_dir', type=str, default='/home/jovyan/in/project/RT_DETR/dataset/coco',
                        help='path to the dataset directory (default: dataset/coco)'
                        'This is the directory that must contains the train2017, val2017, annotations folder')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers (default: 0)')

    parser.add_argument('--val', type=str2bool, default=False,
                        help='if True, only evaluate the model (default: False)')

    parser.add_argument('--amp', type=str2bool, default=True,
                        help='When GPU is available, use Automatic Mixed Precision (default: True)')
    
    parser.add_argument('--ema', type=str2bool, default=True,
                        help='Use Exponential Moving Average (default: True)')

    parser.add_argument('--epoch', type=int, default=100,
                        help='When test-only is False, this is the number of epochs to train (default: 100)')

    parser.add_argument('--model_type', type=str, default='r18vd',
                        choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd'],
                        help='choose the model type (default: r18vd)')

    main()