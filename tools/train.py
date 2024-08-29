from src import zoo
from utils import fit, val
from src.data.coco.coco_dataset import CocoDetection
from src.misc import dist
import argparse

def main():
    args = parser.parse_args()

    dist.init_distributed()
    
    model = zoo.model.r18vd()
    optimizer = zoo.optimizer.rtdetr_r18vd_optimizer(model)

    #TODO There is a slow on a dataset that is not a CocoDetection class, need to fix this
    val_dataset = zoo.rtdetr_val_dataset(dataset_class=CocoDetection)
    val_dataloader = zoo.rtdetr_val_dataloader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.test_only:
        val(model, args.weight_path, val_dataloader=val_dataloader)
    else:
        train_dataloader = zoo.rtdetr_train_dataloader(batch_size=args.batch_size, num_workers=args.num_workers)
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
    parser.add_argument('--weight_path', '-w', type=str, default="output/rtdetr_r18vd_6x_coco/86.pth")

    parser.add_argument('--save_dir', '-save', type=str, default="output/rtdetr_r18vd_6x_coco")

    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--test-only', action='store_true', default=False)

    parser.add_argument('--amp', action='store_true', default=True,
                        help='Automatic Mixed Precision')
    parser.add_argument('--ema', action='store_true', default=True,
                        help='Exponential Moving Average')
    
    parser.add_argument('--epoch', type=int, default=73)

    main()