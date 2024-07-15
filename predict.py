import torch
import matplotlib as mpl
mpl.use('Agg')

from src.data_pretrain import compile_data
from src.data_test import compile_data_test
from src.tools import (SimpleLoss, get_val_info, gen_dx_bx, get_val_info_new, get_val_info_nobev)
from src.model_BEV_TXT import compile_model_lss, compile_model_bevtxt


def iou_predict(args):
    grid_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
    }
    data_aug_conf = {
        'resize_lim': args.resize_lim,
        'final_dim': args.final_dim,
        'rot_lim': args.rot_lim,
        'H': args.H, 'W': args.W,
        'rand_flip': args.rand_flip,
        'bot_pct_lim': args.bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': args.ncams,
    }
    trainloader, valloader = compile_data(args.version, args.dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=args.bsize, nworkers=args.nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if args.gpuid < 0 else torch.device(f'cuda:{args.gpuid}')

    model = compile_model_lss(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    print('loading', args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint), strict=True)
    model.to(device)
    loss_fn = SimpleLoss().cuda(args.gpuid)

    model.eval()
    iou_info, val_loss = get_val_info(model, valloader, loss_fn, device)
    iou_info = str(iou_info)
    print(iou_info)
    print('val_loss: {}'.format(val_loss))

    # Log the val info
    results_txt = './b1_20.txt'
    with open(results_txt, "a") as f:
        f.write('checkpoint:{}'.format(args.checkpoint) + iou_info + '\n'
                + 'val_loss: ' + str(val_loss) + "\n\n")


def bev_txt_pred(args):

    grid_conf = {'xbound': args.xbound, 'ybound': args.ybound,
                 'zbound': args.zbound, 'dbound': args.dbound, }
    data_aug_conf = {
        'resize_lim': args.resize_lim,
        'final_dim': args.final_dim,
        'rot_lim': args.rot_lim,
        'H': args.H, 'W': args.W,
        'rand_flip': args.rand_flip,
        'bot_pct_lim': args.bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': args.ncams,
    }
    testloader = compile_data_test(args.version, args.dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=args.bsize, nworkers=args.nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if args.gpuid < 0 else torch.device(f'cuda:{args.gpuid}')

    model = compile_model_bevtxt(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    if args.checkpoint:
        print('loading', args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=f'cuda:{args.gpuid}'), strict=True)
    model.to(device)

    model.eval()
    iou_info, category_act, category_desc, act_overall, desc_overall, \
    act_mean, desc_mean = get_val_info_new(model, testloader, device)
    iou_info = str(iou_info)
    print(iou_info)
    AD_info = """
                    F1_Action: {0}
                    F1_Description: {1}
                    Action_overall: {2}
                    Description_overall: {3}
                    Action_mean: {4}
                    Description_mean: {5}
                    """.format(category_act, category_desc, act_overall, desc_overall, act_mean, desc_mean)
    print(AD_info)

    # Log the val info
    results_txt = './test.txt'
    with open(results_txt, "a") as f:
        f.write(args.checkpoint + '\n' + iou_info + '\n' + 'F1_info: ' + AD_info + "\n\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch testing")
    # General Setting
    parser.add_argument("--version", default='trainval', help='[exp, mini]')
    parser.add_argument("--dataroot", default="/path/to/the/dataset/")
    parser.add_argument("--nepochs", default=10000, type=int)
    parser.add_argument("--gpuid", default=1, type=int)
    parser.add_argument("--logdir", default='./test-result/', help='path for the log file')
    parser.add_argument("--bsize", default=1, type=int)
    parser.add_argument("--nworkers", default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--wdecay', default=1e-7, type=float, help='weight decay')
    parser.add_argument('--checkpoint', default='/path/to/the/weight/')
    parser.add_argument('--examplef', default='./test', help='file for bev images')
    parser.add_argument('--seg_classes', default=4, help='number of class in segmentation')

    parser.add_argument('--xbound', default=[-50.0, 50.0, 0.5], help='grid configuration')
    parser.add_argument('--ybound', default=[-50.0, 50.0, 0.5], help='grid configuration')
    parser.add_argument('--zbound', default=[-10.0, 10.0, 20.0], help='grid configuration')
    parser.add_argument('--dbound', default=[4.0, 45.0, 1.0], help='grid configuration')
    parser.add_argument('--H', default=900, type=int)
    parser.add_argument('--W', default=1600, type=int)
    parser.add_argument('--resize_lim', default=(0.193, 0.225))
    parser.add_argument('--final_dim', default=(128, 352))
    parser.add_argument('--bot_pct_lim', default=(0.0, 0.22))
    parser.add_argument('--rot_lim', default=(-5.4, 5.4))
    parser.add_argument('--rand_flip', default=True, type=bool)
    parser.add_argument('--ncams', default=6, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # iou_predict(args)
    bev_txt_pred(args)