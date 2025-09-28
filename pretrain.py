import torch
from time import time
import numpy as np
import os
import tqdm
from src.model_baseline import compile_model_lss
from src.data_pretrain import compile_data
from src.tools import SimpleLoss, get_val_info


def train(args):

    max_grad_norm = 5.0
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
    trainloader, valloader = compile_data(
        args.version,
        args.dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=args.bsize,
        nworkers=args.nworkers,
        parser_name="segmentationdata",
    )

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    device = torch.device("cuda")

    model = compile_model_lss(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    if args.checkpoint:
        print("loading", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint), strict=True)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    loss_fn = SimpleLoss().cuda(args.gpuid)

    counter = 0
    best_iou = -float("inf")
    for epoch in range(args.nepochs):
        print("--------------Epoch: {}--------------".format(epoch))
        np.random.seed()
        model.train()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(tqdm.tqdm(trainloader)):
            t0 = time()
            opt.zero_grad()
            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            binimgs = binimgs.to(device)

            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 200 == 0:
                print("Counter{} Train_Loss: {}".format(counter, loss.item()))

        # val_info
        iou_info_raw, val_loss = get_val_info(model, valloader, loss_fn, device)
        iou_info = str(iou_info_raw)
        print(iou_info)
        print("val_loss: {}".format(val_loss))

        # Log the val info
        results_txt = "./logs/pretrain/pretrain_log.txt"
        with open(results_txt, "a") as f:
            f.write("Epoch {}\n".format(epoch) + iou_info + "\n" + "val_loss: " + str(val_loss) + "\n\n")

        # Save the weight (에폭별 저장)
        current_iou = iou_info_raw.compute()[2]
        if current_iou > best_iou:
            best_iou = current_iou
            best_mname = os.path.join(args.logdir, "best_model.pt")
            print("best model confirmed! saving at epoch {}".format(epoch))
            torch.save(model.state_dict(), best_mname)

        mname = os.path.join(args.logdir, "model{}.pt".format(epoch))
        print("saving", mname)
        torch.save(model.state_dict(), mname)
        model.train()

    with open(results_txt, "a") as f:
        f.write("-" * 100 + "\n")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch pre-training")
    # General Setting
    parser.add_argument("--version", default="trainval", help="[trainval, mini]")
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--nepochs", default=30, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--logdir", default="./logs/pretrain/model_weights/", help="path for the log file")
    parser.add_argument("--bsize", default=12, type=int)
    parser.add_argument("--nworkers", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--wdecay", default=1e-7, type=float, help="weight decay")
    parser.add_argument("--checkpoint", default="")
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
    train(args)
