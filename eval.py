import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.distributed
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

from timm.data import Mixup, create_transform
from timm.utils import accuracy, AverageMeter
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch import optim as optim

from torchvision import datasets, transforms

from thop import profile, clever_format

import models
from utils import check_keywords_in_name, create_logger, NativeScalerWithGradNormCount, reduce_tensor

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

# 指定哪几张GPU进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


def get_args_parser():
    parser = argparse.ArgumentParser("Models for image classification", add_help=False)

    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for a single GPU")  # Depend on the task
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")  # Depend on the task
    parser.add_argument('--img_size', type=int, default=224, help="Input image size")
    parser.add_argument('--dataset', type=str, required=True, help="Type of dataset")
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help="Interpolation to resize image (random, bilinear, bicubic)")
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loading threads")
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name')
    parser.add_argument('--resume', type=str, default='', help="Checkpoint to resume")
    parser.add_argument('--num_classes', type=int, default=1000, help="Number of classes")  # Depend on the task
    parser.add_argument('--amp_enable', type=bool, default=True, help="Enable Pytorch automatic mixed precision (amp)")
    parser.add_argument('--output', type=str, default='output', help="Path to output folder")
    parser.add_argument('--tag', type=str, default='eval', help="Tag of experiment")
    parser.add_argument('--print_freq', type=int, default=20, help="Frequency to logging info")
    parser.add_argument('--seed', type=int, default=16, help="Fixed random seed")
    parser.add_argument('--local_rank', type=int, required=True, help="Local rank for DistributedDataParallel")

    args, unparsed = parser.parse_known_args()

    return args


def main(args):

    size = int((256 / 224) * args.img_size)
    val_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=_pil_interp(args.interpolation)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    if args.dataset == 'ImageNet':
        val_dataset = datasets.ImageFolder(args.data_path + '/val', transform=val_transforms)
        args.num_classes = 1000
    elif args.dataset == 'CIFAR100':
        val_dataset = datasets.CIFAR100(root=args.data_path, train=False, transform=val_transforms)
        args.num_classes = 100
    elif args.dataset == 'CIFAR10':
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=val_transforms)
        args.num_classes = 10

    val_sampler = torch.utils.data.DistributedSampler(dataset=val_dataset, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, sampler=val_sampler, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )

    logger.info(f"Creating model:{args.model_name}")
    model = models.__dict__[args.model_name](num_classes=args.num_classes)
    logger.info(f"Using model: {args.model_name}")

    flops, params = profile(model, inputs=(torch.rand([1, 3, 224, 224]), ))
    flops, params = clever_format([flops, params], "%.2f")
    logger.info(f"The number of parameters: {params}")
    logger.info(f"number of GFLOPs: {flops}")

    model.cuda()
    # 先将模型保存到model_without_ddp
    # 使用DDP后, 原模型已经被封装, 运行时进行发布
    # 对于模型的保存, 要么先将模型保存到model_without_ddp, 保存时保存model_without_ddp; 要么就在DDP后, 保存model.module模块
    # 读取checkpoint也需将checkpoint保存到model_without_ddp
    model_without_ddp = model

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    del checkpoint
    torch.cuda.empty_cache()
    acc1, acc5, loss = validate(args, val_loader, model)
    logger.info(f"Accuracy of the model on the {len(val_dataset)} val images: {acc1:.1f}%")

@torch.no_grad()
def validate(args, val_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()  # 开始验证某个batch的时刻
    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp_enable):
            output = model(images)

        # 计算精度和损失
        loss = criterion(output, labels)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), labels.size(0))
        acc1_meter.update(acc1.item(), labels.size(0))
        acc5_meter.update(acc5.item(), labels.size(0))

        # 计算验证一个batch所需的时间
        batch_time.update(time.time() - end)
        end = time.time()  # 充值开始验证某个batch的时刻

        if step % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f"Test:[{step}/{len(val_loader)}] "
                        f"Time:{batch_time.val:.3f}({batch_time.avg:.3f}) "
                        f"Loss:{loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                        f"Acc@1:{acc1_meter.val:.3f}({acc1_meter.avg:.3f}) "
                        f"Acc@5:{acc5_meter.val:.3f}({acc5_meter.avg:.3f}) "
                        f"Mem:{memory_used:.0f}MB")
    logger.info(f"====================> Acc@1: {acc1_meter.avg:.3f}, Acc@5: {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args = get_args_parser()

    # ========================= 分布式训练初始化 =========================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    # windows系统只支持gloo, 在linux系统上推荐使用nccl
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()  # 实现不同进程之间的数据同步

    # ========================= 设置随机种子 =========================
    seed = args.seed + torch.distributed.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True  # 让cudnn内置的auto-tuner自动寻找最适合当前配置的高效算法, 优化运行效率

    # 模型输出的保存位置
    args.output = os.path.join(args.output, args.tag)
    os.makedirs(args.output, exist_ok=True)
    # 创建日志器, 记录训练过程
    logger = create_logger(output_dir=args.output, dist_rank=torch.distributed.get_rank(), name=f"{args.model_name}")

    main(args)
