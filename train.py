import torch

torch.set_float32_matmul_precision("high")
import numpy as np
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.model_BEV_TXT import compile_model_bevtxt
from src.data import compile_data
from src.tools import MultiLoss, get_val_info_new

action_categories = [
    "Move_forward",
    "Stop_slow_down",
    "Turn_left_change_to_left_lane",
    "Turn_right_change_to_right_lane",
]

desc_categories = [
    "Traffic_light_allows",
    "Front_area_is_clear",
    "Solid_line_on_the_left",
    "Solid_line_on_the_right",
    "Front_left_area_is_clear",
    "Back_left_area_is_clear",
    "Front_right_area_is_clear",
    "Back_right_area_is_clear",
]


def train(args, grid_conf, data_aug_conf, max_grad_norm):
    print("[Info] Ready for training...")
    device = torch.device("cuda:0")
    print("[Info] Device: {}".format(device))

    print("[Info] Creating log directory...")
    tb_logdir = os.path.join(args.logdir, "tensorboard")
    if not os.path.exists(tb_logdir):
        os.mkdir(tb_logdir)
    writer = SummaryWriter(tb_logdir)

    print("[Info] Compiling data...")
    trainloader, valloader = compile_data(
        args.version,
        args.dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=args.bsize,
        nworkers=args.nworkers,
    )

    print("[Info] Compiling model, optimizer...")
    model = compile_model_bevtxt(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
        print("[Info] Loaded checkpoint from {}".format(args.checkpoint))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    best_metric = -float("inf")
    print("[Info] Start training...")
    for epoch in range(args.nepochs):
        print("--------------Epoch: {}--------------".format(epoch))
        np.random.seed()
        model.train()

        # Train loss accumulation
        train_loss_sum = 0.0
        train_loss_bev_sum = 0.0
        train_loss_act_sum = 0.0
        train_loss_desc_sum = 0.0
        train_batches = 0

        # Train loop
        pbar = tqdm.tqdm(trainloader, dynamic_ncols=True, ncols=None, desc="Training")
        for imgs, rots, trans, intrins, post_rots, post_trans, binimgs, acts, descs in pbar:
            opt.zero_grad()
            bev_pres, act_pres, desc_pres = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )

            binimgs = binimgs.to(device)[:, -1, :, :]
            acts = acts.to(device)[:, -1, :]
            descs = descs.to(device)[:, -1, :]
            loss, loss_bev, loss_act, loss_desc = MultiLoss(bev_pres, act_pres, desc_pres, binimgs, acts, descs, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            train_loss_sum += loss.item()
            train_loss_bev_sum += loss_bev.item()
            train_loss_act_sum += loss_act.item()
            train_loss_desc_sum += loss_desc.item()
            train_batches += 1

        # Evaluation
        print("[Info] Running eval...")
        (
            iou_info_raw,
            category_act,
            category_desc,
            act_overall,
            desc_overall,
            act_mean,
            desc_mean,
        ) = get_val_info_new(model, valloader, device)
        iou_info = str(iou_info_raw)
        print(iou_info)

        AD_info = ""
        AD_info += "F1_Action: {}\n".format(["{:.3f}".format(x) for x in category_act])
        AD_info += "F1_Description: {}\n".format(["{:.3f}".format(x) for x in category_desc])
        AD_info += "F1_Action_Overall: {:.3f}\n".format(act_overall)
        AD_info += "F1_Description_Overall: {:.3f}\n".format(desc_overall)
        AD_info += "F1_Action_Mean: {:.3f}\n".format(act_mean)
        AD_info += "F1_Description_Mean: {:.3f}\n".format(desc_mean)
        print(AD_info)

        # Saving
        print("[Info] Saving the weight...")
        mname = os.path.join(args.logdir, "model{}.pt".format(epoch))
        torch.save(model.state_dict(), mname)
        current_metric = (act_overall + desc_overall) / 2
        if current_metric > best_metric:
            print("[Info] Best model confirmed! Saving at epoch {}".format(epoch))
            best_metric = current_metric
            best_mname = os.path.join(args.logdir, "best_model.pt")
            torch.save(model.state_dict(), best_mname)

        # Logging
        print("[Info] Logging the val info...")

        # Log the val info
        with open("./logs/train/train_log.txt", "a") as f:
            # Text Logging
            write_str = ""
            write_str += f"Epoch {epoch}\n"
            write_str += f"{iou_info}\n"
            write_str += f"{AD_info}\n"
            write_str += "-" * 100 + "\n"
            f.write(write_str)

            # TensorBoard Logging
            avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0
            avg_train_loss_bev = train_loss_bev_sum / train_batches if train_batches > 0 else 0.0
            avg_train_loss_act = train_loss_act_sum / train_batches if train_batches > 0 else 0.0
            avg_train_loss_desc = train_loss_desc_sum / train_batches if train_batches > 0 else 0.0
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Train_Bev", avg_train_loss_bev, epoch)
            writer.add_scalar("Loss/Train_Act", avg_train_loss_act, epoch)
            writer.add_scalar("Loss/Train_Desc", avg_train_loss_desc, epoch)

            # Action category-wise F1 score logging
            for i, f1_score in enumerate(category_act):
                writer.add_scalar(f"F1_Action/{action_categories[i]}", f1_score, epoch)

            # Description category-wise F1 score logging
            for i, f1_score in enumerate(category_desc):
                writer.add_scalar(f"F1_Description/{desc_categories[i]}", f1_score, epoch)

            # Overall metric logging
            writer.add_scalar("Action_overall", act_overall, epoch)
            writer.add_scalar("Description_overall", desc_overall, epoch)
            writer.add_scalar("Action_mean", act_mean, epoch)
            writer.add_scalar("Description_mean", desc_mean, epoch)

        model.train()

    writer.close()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch training")
    # General Setting
    parser.add_argument("--version", default="trainval", help="[trainval, mini]")
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--nepochs", default=60, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--logdir", default="./logs/train/model_weights/", help="path for the log file")
    parser.add_argument("--bsize", default=6, type=int)
    parser.add_argument("--nworkers", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--wdecay", default=1e-8, type=float, help="weight decay")
    parser.add_argument("--checkpoint", default="./logs/pretrain/model_weights/best_model.pt")
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
    parser.add_argument("--rand_flip", default=False, type=bool)
    parser.add_argument("--ncams", default=6, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

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

    train(args, grid_conf, data_aug_conf, max_grad_norm)
