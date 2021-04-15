from __future__ import print_function
import cv2
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from thop import profile

from utils.config import cfg_mnet
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

cudnn.benchmark = True


def load_normal(load_path):
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def save_image(dets, vis_thres, img_raw, save_folder, img_name, save_all=False):
    count_ = 0
    line = ''
    for b in dets:
        if b[4] < vis_thres:
            continue
        count_ += 1
        line += ' ' + str(int(b[0])) + ' ' + str(int(b[1])) + ' ' + str(int(b[2])) + ' ' + str(int(b[3]))
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image
    name = os.path.join(os.path.join(save_folder, 'pictures'), img_name)
    dirname = os.path.dirname(name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if save_all:
        cv2.imwrite(name, img_raw)
    else:
        if count_ > 0:
            cv2.imwrite(name, img_raw)
    line_write = os.path.split(img_name)[-1] + ' ' + str(count_) + line + '\n'
    return line_write


def run(args):
    # net and load
    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase='test')
    new_state_dict = load_normal(args.trained_model)
    net.load_state_dict(new_state_dict)
    print('Finished loading model!')
    print(net)

    torch.set_grad_enabled(False)

    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    input = torch.randn(1, 3, 270, 480).cuda()
    flops, params = profile(net, inputs=(input,))
    print('flops:', flops, 'params:', params)

    # testing dataset
    with open(args.test_list_dir, 'r') as fr:
        test_dataset = fr.read().split()
    test_dataset.sort()

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    f_ = open(os.path.join(args.save_folder, 'vis_bbox.txt'), 'w')
    net.eval()
    for i, image_path in enumerate(test_dataset):
        #img_name = os.path.split(image_path)[-1]
        img_name = image_path[image_path.find('datasets') + 9:]
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # img_raw = cv2.resize(img_raw, None, fx=1./3, fy=1.0/3, interpolation=cv2.INTER_AREA)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_name = os.path.join(args.save_folder, 'txt', img_name)[:-4] + '.txt'
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d}'
              ' forward_pass_time: {:.4f}s'
              ' misc: {:.4f}s'
              ' img_shape:{:}'.format
              (i + 1, len(test_dataset),
               _t['forward_pass'].average_time,
               _t['misc'].average_time,
               img.shape))

        # save bbox-image
        line_write = save_image(dets, args.vis_thres, img_raw, args.save_folder, img_name, save_all=args.save_image_all)
        f_.write(line_write)
        f_.flush()
    f_.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('--trained_model', type=str, default='./results_train/21-04-14-18-32-21/mobile0.25_150.pth')
    parser.add_argument('--test_list_dir', type=str, default='./data_list/valid_widerface_list.txt')
    parser.add_argument('--save_folder', type=str, default='./results_val-test/widerface')
    # parser.add_argument('--test_list_dir', type=str, default='./data_list/test_san-480_list.txt')
    # parser.add_argument('--save_folder', type=str, default='./results_val-test/san-480')
    parser.add_argument('--save_image_all', default=True)

    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

    parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=500, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.25, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=100, type=int, help='keep_top_k')

    parser.add_argument('--vis_thres', default=0.75, type=float, help='visualization_threshold')

    args = parser.parse_args()

    run(args)
