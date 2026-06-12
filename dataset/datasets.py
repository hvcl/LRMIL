import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


def patient_id_from_wsi(wsi_name: str, dataset_name: Optional[str] = None) -> str:
    if dataset_name and dataset_name.lower() == "bracs":
        return wsi_name
    return "-".join(wsi_name.split("-")[:3])


def label_column(task: str) -> str:
    return "event" if task == "survival" else task


class TeacherMILDataset(Dataset):
    def __init__(self, aggregated_root, csv_path, fold_id=1, split="train", task="subtype", dataset_name=None):
        df = pd.read_csv(csv_path)
        fold_col = f"fold_{fold_id}"
        self.pid_to_split = dict(zip(df["P_ID"], df[fold_col]))
        self.pid_to_label = dict(zip(df["P_ID"], df[label_column(task)]))
        self.samples = []

        for fname in sorted(os.listdir(aggregated_root)):
            if not fname.endswith("_aggregated.pt"):
                continue
            wsi_name = fname.replace("_aggregated.pt", "")
            pid = patient_id_from_wsi(wsi_name, dataset_name)
            if self.pid_to_split.get(pid) != split:
                continue
            self.samples.append(
                {
                    "path": os.path.join(aggregated_root, fname),
                    "wsi_name": wsi_name,
                    "pid": pid,
                    "label": int(self.pid_to_label[pid]),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        obj = torch.load(item["path"], map_location="cpu")
        feats = obj["teacher_final"]
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        coords = obj["coords_20x"]
        return {
            "E20": feats,   
            "coords_20x": coords,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "pid": item["pid"],
            "wsi_name": item["wsi_name"],
        }


class StudentMILDataset(Dataset):
    def __init__(self, student_feats_root, stage1_root, csv_path, fold_id=1, split="train", task="subtype", dataset_name=None):
        df = pd.read_csv(csv_path)
        fold_col = f"fold_{fold_id}"
        self.pid_to_split = dict(zip(df["P_ID"], df[fold_col]))
        self.pid_to_label = dict(zip(df["P_ID"], df[label_column(task)]))
        self.samples = []

        meta_dir = os.path.join(stage1_root, "teacher_meta", f"fold{fold_id}")

        for fname in sorted(f for f in os.listdir(meta_dir) if f.endswith(".pt")):
            wsi_name = fname[:-3]
            pid = patient_id_from_wsi(wsi_name, dataset_name)
            if self.pid_to_split.get(pid) != split:
                continue
            student_path = os.path.join(student_feats_root, f"{wsi_name}.pt")
            if not os.path.exists(student_path):
                continue
            self.samples.append(
                {
                    "wsi_name": wsi_name,
                    "pid": pid,
                    "label": int(self.pid_to_label[pid]),
                    "student_path": student_path,
                    "meta_path": os.path.join(meta_dir, fname),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        student_file = torch.load(item["student_path"], map_location="cpu")
        if isinstance(student_file, dict):
            feats = student_file["feats_5x"]
            coords_5x = student_file["coords_5x"]
        else:
            feats = student_file
            coords_5x = torch.empty(0, 2, dtype=torch.long)
        if feats.dim() > 2:
            feats = feats.view(feats.shape[0], -1)

        meta = torch.load(item["meta_path"], map_location="cpu")
        return {
            "wsi_name": item["wsi_name"],
            "student_feats": feats,
            "coords_5x": coords_5x.squeeze(0),
            "coords_20x": meta["coords_20x"].squeeze(0),
            "teacher_attn": meta["attention"].view(-1),
            "teacher_logits": meta["logits"],
            "teacher_bag": meta["bag_repr"],
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


class PatchKDDataset(Dataset):
    def __init__(
        self,
        feats5x_root,
        aggregated_root,
        stage1_root,
        csv_path,
        fold_id=1,
        split="train",
        task="subtype",
        dataset_name=None,
    ):
        df = pd.read_csv(csv_path)
        self.pid_to_split = dict(zip(df["P_ID"], df[f"fold_{fold_id}"]))
        self.items = []
        self.concat_cache = {}
        self.agg_cache = {}
        self.meta_cache = {}
        self.feats5x_root = feats5x_root

        for wsi_name in sorted(os.listdir(feats5x_root)):
            concat_path = os.path.join(feats5x_root, wsi_name, "level1_images_concat_withcoords.pt")
            if not os.path.exists(concat_path):
                continue
            pid = patient_id_from_wsi(wsi_name, dataset_name)
            if self.pid_to_split.get(pid) != split:
                continue
            agg_path = os.path.join(aggregated_root, f"{wsi_name}_aggregated.pt")
            meta_path = os.path.join(stage1_root, "teacher_meta", f"fold{fold_id}", f"{wsi_name}.pt")
            if not os.path.exists(agg_path) or not os.path.exists(meta_path):
                continue
            agg = torch.load(agg_path, map_location="cpu")
            self.agg_cache[wsi_name] = agg
            self.meta_cache[wsi_name] = torch.load(meta_path, map_location="cpu")
            coords_5x = agg["coords_5x"]
            coord2idx = {(int(x), int(y)): i for i, (x, y) in enumerate(coords_5x.tolist())}
            concat = torch.load(concat_path, map_location="cpu")
            for img_index, coord in enumerate(concat["coords_5x"]):
                key = tuple(map(int, coord.tolist()))
                idx5 = coord2idx.get(key)
                if idx5 is None or len(agg["links_5x_to_20x"][idx5]) == 0:
                    continue
                self.items.append({"wsi_name": wsi_name, "img_index": img_index, "idx_5x": idx5})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wsi = item["wsi_name"]
        if wsi not in self.concat_cache:
            path = os.path.join(self.feats5x_root, wsi, "level1_images_concat_withcoords.pt")
            self.concat_cache[wsi] = torch.load(path, map_location="cpu")
        concat = self.concat_cache[wsi]
        agg = self.agg_cache[wsi]
        meta = self.meta_cache[wsi]
        idx5 = item["idx_5x"]
        links = torch.tensor(agg["links_5x_to_20x"][idx5], dtype=torch.long)
        attn = meta["attention"].view(-1)
        return {
            "img5": concat["images"][item["img_index"]],
            "coords_5x": agg["coords_5x"][idx5],
            "coords_20x": agg["coords_20x"][links],
            "teacher_final": agg["teacher_final"][links],
            "teacher_layers": {layer: value[links] for layer, value in agg["teacher_layers"].items()},
            "teacher_attn": attn[links],
            "wsi_name": wsi,
        }
