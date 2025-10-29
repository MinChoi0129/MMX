import torch
from torch.optim.lr_scheduler import StepLR

torch.set_float32_matmul_precision("high")
import numpy as np
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.model_baseline import compile_model_lss
from src.data_pretrain import compile_data
from src.tools import SimpleLoss, get_val_info


def pretrain(args, grid_conf, data_aug_conf, max_grad_norm):
    print("[Info] Ready for pretraining...")
    device = torch.device("cuda:0")
    print("[Info] Device: {}".format(device))

    print("[Info] Creating log directory...")
    tb_logdir = os.path.join(args.logdir, "tensorboard")
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir, exist_ok=True)
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

    print("[Info] Preparing model, optimizer, scheduler, and loss function...")
    model = compile_model_lss(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    for param in model.encoder.parameters():
        param.requires_grad = True
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = StepLR(opt, step_size=5, gamma=0.1)
    loss_fn = SimpleLoss().to(device)

    best_iou = 0.0
    print("[Info] Start training...")
    for epoch in range(args.nepochs):
        print("--------------Epoch: {}--------------".format(epoch))
        np.random.seed()
        model.train()

        # Train loss accumulation
        train_loss_sum = 0.0
        train_batches = 0

        # Train loop
        pbar = tqdm.tqdm(trainloader, dynamic_ncols=True, ncols=None, desc="Training")
        for imgs, rots, trans, intrins, post_rots, post_trans, binimgs in pbar:
            opt.zero_grad()
            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )

            binimgs = binimgs.to(device)[:, :, :]
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            train_loss_sum += loss.item()
            train_batches += 1

        # Evaluation
        print("[Info] Running eval...")
        iou_info_raw, val_loss = get_val_info(model, valloader, loss_fn, device)
        iou_info = str(iou_info_raw)
        print(iou_info)

        # Saving
        print("[Info] Saving the weight...")
        mname = os.path.join(args.logdir, "model{}.pt".format(epoch))
        torch.save(model.state_dict(), mname)
        current_iou = iou_info_raw.compute()[2].mean().item()
        if current_iou > best_iou:
            print("[Info] Best model confirmed! Saving at epoch {}".format(epoch))
            best_iou = current_iou
            best_mname = os.path.join(args.logdir, "best_model.pt")
            torch.save(model.state_dict(), best_mname)

        # Logging
        print("[Info] Logging the val info...")
        with open("./logs/pretrain/pretrain_log.txt", "a") as f:
            # Text Logging
            write_str = ""
            write_str += f"Epoch {epoch}\n"
            write_str += f"{iou_info}\n"
            write_str += f"val_loss: {val_loss}\n"
            write_str += "-" * 100 + "\n"
            f.write(write_str)

            # TensorBoard Logging
            avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("IoU/Val", current_iou, epoch)
            writer.add_scalar("IoU/Best", best_iou, epoch)

        model.train()
        scheduler.step(epoch)

    writer.close()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch pre-training")
    # General Setting
    parser.add_argument("--version", default="trainval", help="[trainval, mini]")
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--nepochs", default=30, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--logdir", default="./logs/pretrain/model_weights/", help="path for the log file")
    parser.add_argument("--bsize", default=7, type=int)
    parser.add_argument("--nworkers", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")  # 1e-3 이었음
    parser.add_argument("--wdecay", default=1e-4, type=float, help="weight decay")  # 1e-7 이었음
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

    pretrain(args, grid_conf, data_aug_conf, max_grad_norm)
