from __future__ import print_function
import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data import DataLoader

from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.seed_init import rand_seed
from utils.data.dataset import MyDataset, detection_collate
from utils.data.augment import preproc
from utils.load_model import load_normal
from utils.lr_mul import WarmupCosineSchedule
from utils.log import logger_init
from utils.config import cfg_mnet

cudnn.benchmark = True


def main(args):
    # dist init
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # dataset
    rgb_mean = (104, 117, 123)  # bgr order
    dataset = MyDataset(args.txt_path, args.txt_path2, preproc(args.img_size, rgb_mean))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, args.bs, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate,
                            pin_memory=True, sampler=sampler)

    # net and load
    net = RetinaFace(cfg=cfg_mnet)
    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = load_normal(args.resume_net)
        net.load_state_dict(state_dict)
        print('Loading success!')
    net = net.cuda()
    # ddp
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = ddp(net, device_ids=[args.local_rank], find_unused_parameters=True)

    # optimizer and loss
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = WarmupCosineSchedule(optimizer, args.warm_epoch, args.max_epoch, len(dataloader), args.cycles)

    num_classes = 2
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    # priorbox
    priorbox = PriorBox(cfg_mnet, image_size=(args.img_size, args.img_size))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    # save folder
    if args.local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d-%H-%M-%S')
        args.save_folder = os.path.join(args.save_folder, time_str)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        logger = logger_init(args.save_folder)
        logger.info(args)
    else:
        logger = None

    # train
    for i_epoch in range(args.max_epoch):
        net.train()
        dataloader.sampler.set_epoch(i_epoch)
        for i_iter, data in enumerate(dataloader):
            load_t0 = time.time()
            images, targets = data[:2]
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            # forward
            out = net(images)

            # backward
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg_mnet['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print info
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (len(dataloader) * (args.max_epoch - i_epoch) - i_iter))
            if args.local_rank == 0:
                logger.info('Epoch:{}/{} || Iter: {}/{} || '
                        'Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || '
                        'LR: {:.8f} || '
                        'Batchtime: {:.4f} s || '
                        'ETA: {}'.format(
                i_epoch + 1, args.max_epoch, i_iter + 1, len(dataloader),
                loss_l.item(), loss_c.item(), loss_landm.item(),
                optimizer.state_dict()['param_groups'][0]['lr'],
                batch_time, str(datetime.timedelta(seconds=eta))))
        if (i_epoch + 1) % args.save_fre == 0:
            if args.local_rank == 0:
                save_name = 'mobile0.25_' + str(i_epoch + 1) + '.pth'
                torch.save(net.state_dict(), os.path.join(args.save_folder, save_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--txt_path', type=str, default='data_list/train_widerface_list.txt')
    parser.add_argument('--txt_path2', type=str, default=None)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--warm_epoch', type=int, default=5)
    parser.add_argument('--max_epoch', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--cycles', type=float, default=0.5)

    parser.add_argument('--resume_net', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0, help='master rank')
    parser.add_argument('--save_folder', type=str, default='./results_train/')
    parser.add_argument('--save_fre', type=int, default=2)

    args = parser.parse_args()

    rand_seed(0)
    main(args)
