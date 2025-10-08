import matplotlib as mpl

mpl.use("Agg")

import torch

torch.set_float32_matmul_precision("high")

import os
from src.data_test import compile_data_test
from src.tools import get_val_info_new
from src.model_BEV_TXT import compile_model_bevtxt


def bev_txt_pred(args, grid_conf, data_aug_conf):
    print("[Info] Ready for testing...")
    device = torch.device("cuda:0")
    print("[Info] Device: {}".format(device))

    print("[Info] Compiling data...")
    testloader = compile_data_test(
        args.version,
        args.dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=args.bsize,
        nworkers=args.nworkers,
    )

    print("[Info] Compiling model...")
    model = compile_model_bevtxt(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=f"cuda:0"), strict=True)
        print("[Info] Loaded checkpoint from {}".format(args.checkpoint))
    model.to(device)
    model.eval()

    print("[Info] Running test...")
    (
        iou_info,
        category_act,
        category_desc,
        act_overall,
        desc_overall,
        act_mean,
        desc_mean,
    ) = get_val_info_new(model, testloader, device)
    iou_info = str(iou_info)

    AD_info = ""
    AD_info += "F1_Action: {}\n".format(["{:.3f}".format(x) for x in category_act])
    AD_info += "F1_Description: {}\n".format(["{:.3f}".format(x) for x in category_desc])
    AD_info += "F1_Action_Overall: {:.3f}\n".format(act_overall)
    AD_info += "F1_Description_Overall: {:.3f}\n".format(desc_overall)
    AD_info += "F1_Action_Mean: {:.3f}\n".format(act_mean)
    AD_info += "F1_Description_Mean: {:.3f}\n".format(desc_mean)
    print(iou_info, AD_info, sep="\n")

    # Saving
    print("[Info] Saving the test info...")
    txt_filename = "test_log.txt"
    results_path = "./logs/test/"
    results_txt = os.path.join(results_path, txt_filename)

    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    with open(results_txt, "a") as f:
        write_str = ""
        write_str += f"{args.checkpoint}\n"
        write_str += f"{iou_info}\n"
        write_str += f"{AD_info}\n"
        write_str += "-" * 100 + "\n"
        f.write(write_str)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch testing")
    # General Setting
    parser.add_argument("--version", default="trainval", help="[exp, mini]")
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--bsize", default=1, type=int)
    parser.add_argument("--nworkers", default=10, type=int)
    parser.add_argument("--checkpoint", default="./logs/train/model_weights/best_model.pt")
    parser.add_argument("--seg_classes", default=4, help="number of class in segmentation")

    parser.add_argument("--xbound", default=[-50.0, 50.0, 0.5], help="grid configuration")
    parser.add_argument("--ybound", default=[-50.0, 50.0, 0.5], help="grid configuration")
    parser.add_argument("--zbound", default=[-10.0, 10.0, 20.0], help="grid configuration")
    parser.add_argument("--dbound", default=[4.0, 45.0, 1.0], help="grid configuration")
    parser.add_argument("--H", default=900, type=int)
    parser.add_argument("--W", default=1600, type=int)
    parser.add_argument("--resize_lim", default=(0.193, 0.225))
    parser.add_argument("--final_dim", default=(128, 352))
    parser.add_argument("--bot_pct_lim", default=(0.0, 0.22))
    parser.add_argument("--rot_lim", default=(-5.4, 5.4))
    parser.add_argument("--rand_flip", default=True, type=bool)
    parser.add_argument("--ncams", default=6, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    grid_conf = {
        "xbound": args.xbound,
        "ybound": args.ybound,
        "zbound": args.zbound,
        "dbound": args.dbound,
    }

    data_aug_conf = {
        "resize_lim": args.resize_lim,
        "final_dim": args.final_dim,
        "rot_lim": args.rot_lim,
        "H": args.H,
        "W": args.W,
        "rand_flip": args.rand_flip,
        "bot_pct_lim": args.bot_pct_lim,
        "cams": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
        "Ncams": args.ncams,
    }

    bev_txt_pred(args, grid_conf, data_aug_conf)
