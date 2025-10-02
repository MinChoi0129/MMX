import torch
from time import time
import numpy as np
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.model_BEV_TXT import compile_model_bevtxt
from src.data import compile_data
from src.tools import MultiLoss, get_val_info_new


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

    # TensorBoard writer 초기화
    tb_logdir = os.path.join(args.logdir, "tensorboard")
    if not os.path.exists(tb_logdir):
        os.mkdir(tb_logdir)
    writer = SummaryWriter(tb_logdir)

    device = torch.device("cpu") if args.gpuid < 0 else torch.device(f"cuda:{args.gpuid}")

    model = compile_model_bevtxt(args.bsize, grid_conf, data_aug_conf, outC=args.seg_classes)
    if args.checkpoint:
        print("loading", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    best_metric = -float("inf")
    for epoch in range(args.nepochs):
        print("--------------Epoch: {}--------------".format(epoch))
        np.random.seed()
        model.train()

        # Train loss 누적을 위한 변수
        train_loss_sum = 0.0
        train_batches = 0

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, acts, descs) in enumerate(
            tqdm.tqdm(trainloader, dynamic_ncols=True, ncols=None, desc="Training")
        ):
            opt.zero_grad()
            bev_pres, act_pres, desc_pres = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            binimgs = binimgs.to(device)
            acts = acts.to(device)
            descs = descs.to(device)

            loss = MultiLoss(bev_pres, act_pres, desc_pres, binimgs, acts, descs, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            # Train loss 누적
            train_loss_sum += loss.item()
            train_batches += 1

        # val_info
        iou_info, category_act, category_desc, act_overall, desc_overall, act_mean, desc_mean = get_val_info_new(
            model, valloader, device
        )
        iou_info = str(iou_info)
        print(iou_info)
        AD_info = """
                F1_Action: {0}
                F1_Description: {1}
                Action_overall: {2}
                Description_overall: {3}
                Action_mean: {4}
                Description_mean: {5}
                """.format(
            category_act, category_desc, act_overall, desc_overall, act_mean, desc_mean
        )
        print(AD_info)

        # Train loss 평균 계산
        avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0

        # Log the val info
        results_txt = "./logs/train/train_log.txt"
        with open(results_txt, "a") as f:
            f.write("Epoch {}\n".format(epoch) + iou_info + "\n" + "F1_info: " + AD_info + "\n\n")

        # Save the weight
        current_metric = (act_overall + desc_overall) / 2
        if current_metric > best_metric:
            best_metric = current_metric
            best_mname = os.path.join(args.logdir, "best_model.pt")
            print("best model confirmed! saving at epoch {}".format(epoch))
            torch.save(model.state_dict(), best_mname)

        # TensorBoard 로깅
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Action 카테고리별 F1 점수 기록
        action_categories = [
            "Move_forward",
            "Stop_slow_down",
            "Turn_left_change_to_left_lane",
            "Turn_right_change_to_right_lane",
        ]
        for i, f1_score in enumerate(category_act):
            writer.add_scalar(f"F1_Action/{action_categories[i]}", f1_score, epoch)

        # Description 카테고리별 F1 점수 기록
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
        for i, f1_score in enumerate(category_desc):
            writer.add_scalar(f"F1_Description/{desc_categories[i]}", f1_score, epoch)

        # 전체 메트릭 기록
        writer.add_scalar("Action_overall", act_overall, epoch)
        writer.add_scalar("Description_overall", desc_overall, epoch)
        writer.add_scalar("Action_mean", act_mean, epoch)
        writer.add_scalar("Description_mean", desc_mean, epoch)

        mname = os.path.join(args.logdir, "model{}.pt".format(epoch))
        print("saving", mname)
        torch.save(model.state_dict(), mname)
        model.train()

    with open(results_txt, "a") as f:
        f.write("-" * 100 + "\n")

    # TensorBoard writer 종료
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
    parser.add_argument(
        "--bsize", default=8, type=int
    )  # 10 for b0/b1; 9 for b2; 8 for b3; 6 for b4; 4 for b5; 3 for b6; 2 for b7
    parser.add_argument("--nworkers", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--wdecay", default=1e-8, type=float, help="weight decay")
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
    parser.add_argument("--rand_flip", default=False, type=bool)
    parser.add_argument("--ncams", default=6, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
