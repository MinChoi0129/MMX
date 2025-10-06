import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from src.datasplit_npre import create_splits_scenes

# from nuscenes.utils.splits import create_splits_scenes # For make BEV GT

from nuscenes.utils.data_classes import Box
from glob import glob
from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, data_root):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.dataroot = data_root

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get("sample_data", rec["data"]["CAM_FRONT"])
        imgname = os.path.join(self.nusc.dataroot, sampimg["filename"])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f"{d2}/{d1}/{d0}/{di}/{fi}"

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print("adjusting nuscenes file paths")
            fs = glob(os.path.join(self.nusc.dataroot, "samples/*/samples/CAM*/*.jpg"))
            fs += glob(os.path.join(self.nusc.dataroot, "samples/*/samples/LIDAR_TOP/*.pcd.bin"))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f"samples/{di}/{fi}"] = fname
            fs = glob(os.path.join(self.nusc.dataroot, "sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin"))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f"sweeps/{di}/{fi}"] = fname
            for rec in self.nusc.sample_data:
                if rec["channel"] == "LIDAR_TOP" or (
                    rec["is_key_frame"] and rec["channel"] in self.data_aug_conf["cams"]
                ):
                    rec["filename"] = info[rec["filename"]]

    def get_scenes(self):
        # filter by scene split
        split = {
            "v1.0-trainval": {True: "train", False: "val"},
            "v1.0-mini": {True: "mini_train", False: "mini_val"},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    def sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get("sample_data", rec["data"][cam])
            # path for the image
            imgname = os.path.join(self.nusc.dataroot, samp["filename"])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
            intrin = torch.Tensor(sens["camera_intrinsic"])
            rot = torch.Tensor(Quaternion(sens["rotation"]).rotation_matrix)
            tran = torch.Tensor(sens["translation"])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
        )

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec, dataroot):
        egopose = self.nusc.get("ego_pose", self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"])
        trans = -np.array(egopose["translation"])
        rot = Quaternion(egopose["rotation"]).inverse
        img = np.zeros((self.nx[0], self.nx[1]))

        # load the map and lines
        map_root = os.path.join(dataroot, "local_binmap")
        map_name = rec["token"] + ".npy"
        map_np = np.load(os.path.join(map_root, map_name))
        map_np = np.fliplr(map_np)
        map_np = np.rot90(map_np, 1).astype(float)
        img = img + map_np

        # load the vehicles
        for tok in rec["anns"]:
            inst = self.nusc.get("sample_annotation", tok)
            # add category for lyft
            if not inst["category_name"].split(".")[0] == "vehicle":
                continue
            box = Box(inst["translation"], inst["size"], Quaternion(inst["rotation"]))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round((pts - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        img = img.astype(int)
        img = torch.Tensor(img)
        img = img.long()
        return img

    def choose_cams(self):
        assert self.data_aug_conf["Ncams"] == 6
        cams = self.data_aug_conf["cams"]
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]
        dataroot = self.dataroot

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec, dataroot)
        # binimg = 1  # For make BEV GT

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    # def __getitem__(self, index):
    #     rec = self.ixes[index]
    #     dataroot = self.dataroot

    #     cams = self.choose_cams()
    #     imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
    #     binimg = self.get_binimg(rec, dataroot)
    #     # binimg = 1 # For make BEV GT

    #     return imgs, rots, trans, intrins, post_rots, post_trans, binimg

    def _collect_prev_frames(self, rec, num_needed):
        """
        현재 rec(=t0)에서 시작해 prev 포인터를 따라 최대 num_needed-1개까지
        과거 프레임을 모은 뒤, [..., t-2, t-1, t0] 순서로 반환.
        장면 시작부라 부족하면 가장 오래된 프레임을 반복(pad)해서 길이를 맞춘다.
        """
        frames = [rec]  # [t0]
        cur = rec
        # prev 따라가며 과거 프레임 추가 (t-1, t-2, ...)
        for _ in range(num_needed - 1):
            prev_tok = cur.get("prev", "")
            if not prev_tok:
                break
            prev_rec = self.nusc.get("sample", prev_tok)
            # 장면 경계 넘어가면 중단
            if prev_rec["scene_token"] != rec["scene_token"]:
                break
            frames.append(prev_rec)
            cur = prev_rec

        # frames = [t0, t-1, (t-2), (t-3)] 형태 → 시간 오름차순으로 뒤집어 [t-3..t0]
        frames = frames[::-1]

        # 부족하면 가장 오래된 프레임을 반복해서 앞쪽에 pad
        if len(frames) < num_needed:
            pad_count = num_needed - len(frames)
            frames = [frames[0]] * pad_count + frames

        # 길이를 정확히 맞춤
        return frames[-num_needed:]

    def __getitem__(self, index):
        """
        반환 형태:
          imgs:       [T, 6, 3, H, W]
          rots:       [T, 6, 3, 3]
          trans:      [T, 6, 3]
          intrins:    [T, 6, 3, 3]
          post_rots:  [T, 6, 3, 3]
          post_trans: [T, 6, 3]
          binimgs:    [T, nx[0], nx[1]]
        """
        rec_t0 = self.ixes[index]
        dataroot = self.dataroot
        cams = self.choose_cams()

        # ... ~ t0 프레임 수집
        rec_list = self._collect_prev_frames(rec_t0, num_needed=3)  # [..., t-2, t-1, t0]

        imgs_seq = []
        rots_seq = []
        trans_seq = []
        intrins_seq = []
        post_rots_seq = []
        post_trans_seq = []
        binimgs_seq = []

        for r in rec_list:
            imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(r, cams)
            binimg = self.get_binimg(r, dataroot)

            imgs_seq.append(imgs)  # [6, 3, H, W]
            rots_seq.append(rots)  # [6, 3, 3]
            trans_seq.append(trans)  # [6, 3]
            intrins_seq.append(intrins)  # [6, 3, 3]
            post_rots_seq.append(post_rots)  # [6, 3, 3]
            post_trans_seq.append(post_trans)  # [6, 3]
            binimgs_seq.append(binimg)  # [nx0, nx1]

        # 시간축으로 스택: T
        imgs_seq = torch.stack(imgs_seq, dim=0)  # [T, 6, 3, H, W]
        rots_seq = torch.stack(rots_seq, dim=0)  # [T, 6, 3, 3]
        trans_seq = torch.stack(trans_seq, dim=0)  # [T, 6, 3]
        intrins_seq = torch.stack(intrins_seq, dim=0)  # [T, 6, 3, 3]
        post_rots_seq = torch.stack(post_rots_seq, dim=0)  # [T, 6, 3, 3]
        post_trans_seq = torch.stack(post_trans_seq, dim=0)  # [T, 6, 3]
        binimgs_seq = torch.stack(binimgs_seq, dim=0)  # [T, nx0, nx1]

        return (
            imgs_seq,
            rots_seq,
            trans_seq,
            intrins_seq,
            post_rots_seq,
            post_trans_seq,
            binimgs_seq,
        )


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, nworkers, parser_name):
    nusc = NuScenes(version="v1.0-{}".format(version), dataroot=os.path.join(dataroot, version), verbose=False)
    parser = {
        "vizdata": VizData,
        "segmentationdata": SegmentationData,
    }[parser_name]
    traindata = parser(
        nusc,
        is_train=True,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        data_root=os.path.join(dataroot, version),
    )
    valdata = parser(
        nusc,
        is_train=False,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        data_root=os.path.join(dataroot, version),
    )

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=bsz,
        shuffle=True,
        num_workers=nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init,
        pin_memory=True,
    )

    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=bsz,
        shuffle=False,
        num_workers=nworkers,
        drop_last=True,
        pin_memory=True,
    )

    return trainloader, valloader
