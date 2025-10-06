import os
import time
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
import cv2
import torch.nn.functional as F
from sklearn.metrics import f1_score
import json


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec["data"]["LIDAR_TOP"]
    ref_sd_rec = nusc.get("sample_data", ref_sd_token)
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_time = 1e-6 * ref_sd_rec["timestamp"]

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec["data"]["LIDAR_TOP"]
    current_sd_rec = nusc.get("sample_data", sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec["filename"]))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
        global_from_car = transform_matrix(
            current_pose_rec["translation"], Quaternion(current_pose_rec["rotation"]), inverse=False
        )

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get("calibrated_sensor", current_sd_rec["calibrated_sensor_token"])
        car_from_current = transform_matrix(
            current_cs_rec["translation"], Quaternion(current_cs_rec["rotation"]), inverse=False
        )

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec["timestamp"]
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec["prev"] == "":
            break
        else:
            current_sd_rec = nusc.get("sample_data", current_sd_rec["prev"])

    return points


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera"""
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N"""
    return (pts[2] > 0) & (pts[0] > 1) & (pts[0] < W - 1) & (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor(
        [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
    )


def img_transform(img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose(
    (
        NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
    )
)


normalize_img = torchvision.transforms.Compose(
    (
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    # print(dx, bx, nx)
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        class_weights = [1, 10, 5, 10]
        weight = torch.FloatTensor(class_weights).cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def MultiLoss(bev_pre, act_pre, desc_pre, bev_gt, act_gt, desc_gt, args):
    # bevs_weights = [0.000001, 0.00001, 0.000005, 0.00001]
    bevs_weights = [1, 10, 5, 10]
    bevs_weights = torch.FloatTensor(bevs_weights).cuda(args.gpuid)
    bevloss_fn = torch.nn.CrossEntropyLoss(weight=bevs_weights).cuda(args.gpuid)
    loss_bev = bevloss_fn(bev_pre, bev_gt)
    act_pre = act_pre.cuda(args.gpuid)
    desc_pre = desc_pre.cuda(args.gpuid)

    # act_weights = [20, 250, 250, 250]  # for act
    act_weights = [1, 5, 5, 5]
    # desc_weights = [20, 400, 400, 400, 20, 20, 20, 20]  # for desc
    desc_weights = [1, 5, 5, 5, 1, 1, 1, 1]
    w1 = torch.FloatTensor(act_weights).cuda(args.gpuid)
    w2 = torch.FloatTensor(desc_weights).cuda(args.gpuid)
    loss_act = F.binary_cross_entropy_with_logits(act_pre, act_gt, weight=w1)
    loss_desc = F.binary_cross_entropy_with_logits(desc_pre, desc_gt, weight=w2)

    loss_all = loss_bev + loss_act + loss_desc
    return loss_all


def MultiLoss_nobev(act_pre, desc_pre, bev_gt, act_gt, desc_gt, args):

    act_pre = act_pre.cuda(args.gpuid)
    desc_pre = desc_pre.cuda(args.gpuid)
    act_weights = [20, 250, 250, 250]  # for act
    desc_weights = [20, 400, 400, 400, 20, 20, 20, 20]  # for desc
    w1 = torch.FloatTensor(act_weights).cuda(args.gpuid)
    w2 = torch.FloatTensor(desc_weights).cuda(args.gpuid)
    loss_act = F.binary_cross_entropy_with_logits(act_pre, act_gt, weight=w1)
    loss_desc = F.binary_cross_entropy_with_logits(desc_pre, desc_gt, weight=w2)

    loss_all = loss_act + loss_desc
    return loss_all


def get_val_info(model, valloader, loss_fn, device, use_tqdm=True) -> tuple:
    model.eval()
    confmat = ConfusionMatrix(4)
    total_loss = 0.0
    print("running eval...")
    loader = tqdm(valloader, dynamic_ncols=True, ncols=None, desc="Validation") if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(
                allimgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            # binimgs = binimgs.to(device)
            binimgs = binimgs.to(device)[:, -1, :, :]

            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]
            confmat.update(binimgs.flatten(), preds.argmax(1).flatten())
        confmat.reduce_from_all_processes()

    model.train()
    return confmat, total_loss


def visualize_bev_gt(bev_gt: torch.Tensor):
    import numpy as np, matplotlib.pyplot as plt, time, os

    if not os.path.exists("bev_prediction_images"):
        os.makedirs("bev_prediction_images")

    bev_np = bev_gt[0].cpu().numpy()  # [200, 200]
    plt.imsave(f"bev_prediction_images/bev_gt_map_{time.time()}.png", bev_np, cmap="viridis")


def get_val_info_new(model, valloader, device, use_tqdm=True, act_num=4, desc_num=8):
    model.eval()
    confmat = ConfusionMatrix(4)
    print("running eval...")
    loader = tqdm(valloader, dynamic_ncols=True, ncols=None, desc="Validation") if use_tqdm else valloader
    with torch.no_grad():
        targets_acts = []
        targets_desc = []
        output_acts = []
        output_desc = []
        act_category = [0.0] * 4
        desc_category = [0.0] * 8
        inference_times = []
        is_time_measure = False

        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs, acts_gt, descs_gt = batch
            # visualize_bev_gt(binimgs.to(device))

            if is_time_measure:
                # 정확한 추론 시간 측정
                torch.cuda.synchronize()
                start_time = time.time()
                bev_pres, act_pres, desc_pres = model(
                    allimgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                )
                torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
            else:
                bev_pres, act_pres, desc_pres = model(
                    allimgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                )
            binimgs = binimgs.to(device)
            bev_pres = bev_pres.to(device)
            act_pres = act_pres.to(device)
            desc_pres = desc_pres.to(device)
            predict_act = torch.sigmoid(act_pres) > 0.5
            predict_desc = torch.sigmoid(desc_pres) > 0.5
            predict_act = predict_act.cpu().numpy()
            predict_desc = predict_desc.cpu().numpy()
            acts_gt_numpy = acts_gt.cpu().numpy()
            descs_gt_numpy = descs_gt.cpu().numpy()

            targets_acts.append(acts_gt_numpy)
            output_acts.append(predict_act)
            targets_desc.append(descs_gt_numpy)
            output_desc.append(predict_desc)

            confmat.update(binimgs.flatten(), bev_pres.argmax(1).flatten())

        confmat.reduce_from_all_processes()

        targets_desc = List2List(targets_desc)
        output_desc = List2List(output_desc)
        targets_acts = List2List(targets_acts)
        output_acts = List2List(output_acts)
        # print(output_acts)

        for i in range(act_num):
            act_category[i] = f1_score(targets_acts[i::act_num], output_acts[i::act_num])

        for i in range(desc_num):
            desc_category[i] = f1_score(targets_desc[i::desc_num], output_desc[i::desc_num])
        f1_overall_act = f1_score(targets_acts, output_acts, average="macro")
        f1_overall_desc = f1_score(targets_desc, output_desc, average="macro")

        # 추론 시간 통계 계산 및 출력
        if inference_times:
            import numpy as np

            times_array = np.array(inference_times)
            mean_time = np.mean(times_array) * 1000
            std_time = np.std(times_array) * 1000
            min_time = np.min(times_array) * 1000
            max_time = np.max(times_array) * 1000
            print(
                f"Inference Time Stats in milliseconds: mean={mean_time:.4f}ms, std={std_time:.4f}ms, min={min_time:.4f}ms, max={max_time:.4f}ms"
            )

    model.train()
    return (
        confmat,
        act_category,
        desc_category,
        f1_overall_act,
        f1_overall_desc,
        np.mean(act_category),
        np.mean(desc_category),
    )


def get_val_info_nobev(model, valloader, device, use_tqdm=True, act_num=4, desc_num=8):
    model.eval()
    print("running eval...")
    loader = tqdm(valloader, dynamic_ncols=True, ncols=None, desc="Validation") if use_tqdm else valloader
    with torch.no_grad():
        targets_acts = []
        targets_desc = []
        output_acts = []
        output_desc = []
        act_category = [0.0] * 4
        desc_category = [0.0] * 8

        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs, acts_gt, descs_gt = batch
            act_pres, desc_pres = model(
                allimgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            binimgs = binimgs.to(device)
            act_pres = act_pres.to(device)
            desc_pres = desc_pres.to(device)
            predict_act = torch.sigmoid(act_pres) > 0.5
            predict_desc = torch.sigmoid(desc_pres) > 0.5
            predict_act = predict_act.cpu().numpy()
            predict_desc = predict_desc.cpu().numpy()
            acts_gt_numpy = acts_gt.cpu().numpy()
            descs_gt_numpy = descs_gt.cpu().numpy()

            targets_acts.append(acts_gt_numpy)
            output_acts.append(predict_act)
            targets_desc.append(descs_gt_numpy)
            output_desc.append(predict_desc)

            # confmat.update(binimgs.flatten(), bev_pres.argmax(1).flatten())

        # confmat.reduce_from_all_processes()

        targets_desc = List2List(targets_desc)
        output_desc = List2List(output_desc)
        targets_acts = List2List(targets_acts)
        output_acts = List2List(output_acts)

        for i in range(act_num):
            act_category[i] = f1_score(targets_acts[i::act_num], output_acts[i::act_num])

        for i in range(desc_num):
            desc_category[i] = f1_score(targets_desc[i::desc_num], output_desc[i::desc_num])
        f1_overall_act = f1_score(targets_acts, output_acts, average="macro")
        f1_overall_desc = f1_score(targets_desc, output_desc, average="macro")

    model.train()
    return act_category, desc_category, f1_overall_act, f1_overall_desc, np.mean(act_category), np.mean(desc_category)


def List2List(List):
    Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
    Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])
    Arr = np.vstack((Arr1, Arr2))

    return [i for item in Arr for i in item]


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array(
        [
            [-4.084 / 2.0 + 0.5, W / 2.0],
            [4.084 / 2.0 + 0.5, W / 2.0],
            [4.084 / 2.0 + 0.5, -W / 2.0],
            [-4.084 / 2.0 + 0.5, -W / 2.0],
        ]
    )
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], "#76b900")


def get_nusc_maps(map_folder):
    nusc_maps = {
        map_name: NuScenesMap(dataroot=map_folder, map_name=map_name)
        for map_name in [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]
    }
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get("ego_pose", nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"])
    map_name = scene2map[nusc.get("scene", rec["scene_token"])["name"]]

    rot = Quaternion(egopose["rotation"]).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose["translation"][0], egopose["translation"][1], np.cos(rot), np.sin(rot)])

    poly_names = ["road_segment", "lane"]
    line_names = ["road_divider", "lane_divider"]
    lmap = get_local_map(nusc_maps[map_name], center, 50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap["road_divider"]:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap["lane_divider"]:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159.0 / 255.0, 0.0, 1.0), alpha=0.5)


def save_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get("ego_pose", nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"])
    map_name = scene2map[nusc.get("scene", rec["scene_token"])["name"]]

    rot = Quaternion(egopose["rotation"]).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose["translation"][0], egopose["translation"][1], np.cos(rot), np.sin(rot)])

    poly_names = ["road_segment", "lane"]
    line_names = ["road_divider", "lane_divider"]
    lmap = get_local_map(nusc_maps[map_name], center, 50.0, poly_names, line_names)
    backg = np.zeros((200, 200))
    # backg = np.zeros((300, 300)) # for polar

    for name in poly_names:
        for la in lmap[name]:
            pts = np.round((la - bx) / dx).astype(np.int32)
            cv2.fillPoly(backg, [pts], 2)
    for la in lmap["road_divider"]:
        pts = np.round((la - bx) / dx).astype(np.int32)
        cv2.fillPoly(backg, [pts], 3)
    for la in lmap["lane_divider"]:
        pts = np.round((la - bx) / dx).astype(np.int32)
        cv2.fillPoly(backg, [pts], 3)

    return backg.astype(int)
    # cv2.imshow("Image", backg)
    # cv2.waitKey(0)


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords, layer_names=layer_names, mode="intersect")
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == "drivable_area":
                polygon_tokens = poly_record["polygon_tokens"]
            else:
                polygon_tokens = [poly_record["polygon_token"]]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record["token"]

            line = nmap.extract_line(record["line_token"])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(np.array([xs, ys]).T)

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        class_labels = ["background", "vehicle", "road", "lane"]
        result = ""
        result += "Global Accuracy: {:.3f}\n".format(acc_global.item())
        result += "Per-Class Accuracy : "
        acc_strs = ["{}: {:.3f}".format(label, acc_val.item()) for label, acc_val in zip(class_labels, acc)]
        result += ", ".join(acc_strs) + "\n"
        result += "Per-Class IoU : "
        iu_strs = ["{}: {:.3f}".format(label, iu_val.item()) for label, iu_val in zip(class_labels, iu)]
        result += ", ".join(iu_strs) + "\n"
        result += "Mean IoU (mIoU): {:.3f}\n".format(iu.mean().item())
        result += "Mean IoU (mIoU w/o background): {:.3f}\n".format(iu[1:].mean().item())
        return result
