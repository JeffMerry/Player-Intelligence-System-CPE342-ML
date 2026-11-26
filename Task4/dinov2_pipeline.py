"""Task 4 multi-architecture training and ensemble evaluation.

Importing this module does not download weights, train, or write experiment outputs.
The notebook explicitly calls development, optional refit, and export separately.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

CLASS_NAMES = ("7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD = (123, 116, 104)


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    family: str
    seed: int
    micro_batch: int
    llrd: float = 1.0
    revision: str | None = None
    weights_sha256: str | None = None


# Pin DINO model identities, revisions and checksums for reproducible weight loading.
MODEL_SPECS = {
    "resnet18": ModelSpec("resnet18", "cnn", 342, 8),
    "convnext_tiny": ModelSpec("convnext_tiny", "cnn", 2345, 2),
    "swin_t": ModelSpec("swin_t", "cnn", 2343, 2),
    "dinov2_small": ModelSpec(
        "vit_small_patch14_reg4_dinov2.lvd142m", "dino", 2348, 2, .75,
        "c04b5193082a8d5b0c4856c7937384a48136c5de",
        "dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d",
    ),
    "dinov2_base": ModelSpec(
        "vit_base_patch14_reg4_dinov2.lvd142m", "dino", 2350, 1, .75,
        "3b06466a5548c52b8b98822e1390987695fcbf82",
        "c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006",
    ),
    "dinov2_large": ModelSpec(
        "vit_large_patch14_reg4_dinov2.lvd142m", "dino", 2352, 1, .86,
        "f3c408e77602bb412aa65fb03dfa0d5f95cb3832",
        "c893d72294d4c327e631ff92f428dbc14c4f93cb5581b6c5f9d89bb5d17def27",
    ),
}


@dataclass
class PipelineConfig:
    model_names: tuple[str, ...] = (
        "resnet18", "convnext_tiny", "swin_t", "dinov2_small", "dinov2_base"
    )
    image_size: tuple[int, int] = (224, 448)  # height, width
    epochs: int = 8
    effective_batch: int = 32
    backbone_lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = .05
    drop_path: float = .1
    warmup_epochs: int = 1
    num_workers: int = 0  # Windows Jupyter; also works on Kaggle
    gradient_checkpointing: bool = True
    allow_large: bool = False
    batch_overrides: dict[str, int] = field(default_factory=dict)
    selection_seed: int = 343
    selection_folds: int = 5
    flip_gain: float = .001
    minimum_gain: float = .002
    bootstrap_samples: int = 2000
    bootstrap_pmax: float = .10
    maximum_hours_per_model: float = 10.0

    def validate(self):
        if not self.model_names or len(set(self.model_names)) != len(self.model_names):
            raise ValueError("model_names must be nonempty and unique")
        if set(self.model_names) - MODEL_SPECS.keys():
            raise ValueError("Unknown model name")
        if "resnet18" not in self.model_names:
            raise ValueError("Include resnet18 as a reproducible reference")
        if "dinov2_large" in self.model_names and not self.allow_large:
            raise ValueError("DINOv2-L is opt-in: set allow_large=True explicitly")
        if len(self.image_size) != 2 or any(d < 32 or d % 14 or d % 32 for d in self.image_size):
            raise ValueError("Image dimensions must be multiples of both 14 and 32")
        if self.epochs < 1 or not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("Invalid epochs/warmup")
        if self.selection_folds < 2 or self.bootstrap_samples < 1 or self.num_workers < 0:
            raise ValueError("Invalid split/bootstrap/workers configuration")
        if self.maximum_hours_per_model <= 0:
            raise ValueError("Time budget must be positive")
        for name in self.model_names:
            batch = self.batch_overrides.get(name, MODEL_SPECS[name].micro_batch)
            if batch < 1 or self.effective_batch < batch or self.effective_batch % batch:
                raise ValueError(f"effective_batch must be a multiple of micro batch: {name}")
        if not 0 < self.bootstrap_pmax < 1 or self.minimum_gain < 0 or self.flip_gain < 0:
            raise ValueError("Invalid fixed selection gates")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class Partitions:
    data_dir: Path
    train: pd.DataFrame
    test: pd.DataFrame
    fit: pd.DataFrame
    selection: pd.DataFrame
    outer: pd.DataFrame
    identity: dict


def prepare_partitions(data_dir, split_path, metadata_path, cfg):
    """Keep the existing outer split; choose architectures/TTA only inside its fit side."""
    cfg.validate()
    data_dir, split_path = Path(data_dir), Path(split_path)
    frames = [pd.read_csv(data_dir / f"{split}.csv", dtype=str, keep_default_na=False)
              for split in ("train", "test")]
    train, test = frames
    for frame in frames:
        if not {"id", "file_name", "label"} <= set(frame):
            raise ValueError("Expected id,file_name,label columns")
        if not frame.id.is_unique or frame.id.eq("").any():
            raise ValueError("IDs must be unique and nonempty")
        for value in frame.file_name:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts or ":" in value or "\\" in value:
                raise ValueError(f"Unsafe filename: {value}")
    if set(train.label) != set(map(str, range(5))):
        raise ValueError("Expected numeric train labels 0..4")
    # Test labels, even if supplied accidentally, are never retained for modelling.
    test = test.drop(columns="label")
    manifest = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(manifest[["id", "file_name", "label"]], train)
    if not {"duplicate_group", "subset"} <= set(manifest):
        raise ValueError("Saved split requires duplicate_group and subset")
    if manifest.duplicate_group.eq("").any() or set(manifest.subset) != {"fit", "validation"}:
        raise ValueError("Invalid saved grouping/split")
    if manifest.groupby("duplicate_group").subset.nunique().max() != 1:
        raise ValueError("A group crosses fit/outer")
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if file_sha256(split_path) != meta["split_manifest_sha256"]:
        raise ValueError("Split hash mismatch")
    if file_sha256(data_dir / "train.csv") != meta["train_csv_sha256"]:
        raise ValueError("Train CSV changed since data audit")
    dev = manifest.loc[manifest.subset.eq("fit")].reset_index(drop=True)
    outer = manifest.loc[manifest.subset.eq("validation")].reset_index(drop=True)
    if len(dev) != meta["fit_images"] or len(outer) != meta["validation_images"]:
        raise ValueError("Saved split counts do not match")
    groups_per_class = dev.groupby("label").duplicate_group.nunique()
    if groups_per_class.min() < cfg.selection_folds:
        raise ValueError("Too few independent groups for the inner selection split")
    splitter = StratifiedGroupKFold(n_splits=cfg.selection_folds, shuffle=True,
                                    random_state=cfg.selection_seed)
    a, b = next(splitter.split(dev, dev.label, dev.duplicate_group))
    fit, selection = dev.iloc[np.sort(a)].copy(), dev.iloc[np.sort(b)].copy()
    for frame in (fit, selection, outer):
        if set(frame.label) != set(map(str, range(5))):
            raise ValueError("All partitions must contain five classes")
    for left, right in itertools.combinations((fit, selection, outer), 2):
        if set(left.duplicate_group) & set(right.duplicate_group):
            raise ValueError("Duplicate group leakage")
    for split, frame in (("train", train), ("test", test)):
        missing = [name for name in frame.file_name if not (data_dir / split / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {split} images: {missing[:5]}")
    inner = pd.concat([fit.assign(partition="fit"), selection.assign(partition="selection"),
                       outer.assign(partition="outer")], ignore_index=True)
    identity = {
        "train_csv_sha256": file_sha256(data_dir / "train.csv"),
        "test_csv_sha256": file_sha256(data_dir / "test.csv"),
        "split_manifest_sha256": file_sha256(split_path),
        "partition_sha256": hashlib.sha256(inner.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "outer_note": "Existing development holdout has been inspected/reused; not a new independent test.",
    }
    return Partitions(data_dir, manifest, test, fit.reset_index(drop=True),
                      selection.reset_index(drop=True), outer, identity)


class Letterbox:
    def __init__(self, size):
        self.height, self.width = size

    def __call__(self, image):
        resized = ImageOps.contain(image, (self.width, self.height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.width, self.height), PAD)
        canvas.paste(resized, ((self.width - resized.width) // 2, (self.height - resized.height) // 2))
        return canvas


def image_transform(cfg, training=False, flip=False):
    operations = [Letterbox(cfg.image_size)]
    if flip:
        operations.append(T.RandomHorizontalFlip(p=1.0))
    if training:
        operations += [T.RandomHorizontalFlip(.5),
                       T.RandomAffine(10, translate=(.05, .05), scale=(.9, 1.1), fill=PAD),
                       T.ColorJitter(.3, .3, .3, .05), T.RandomGrayscale(.05)]
    operations += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if training:
        operations.append(T.RandomErasing(p=.25, scale=(.02, .1), value=0))
    return T.Compose(operations)


class ImageDataset(Dataset):
    def __init__(self, frame, image_root, transform, with_labels=True):
        self.frame = frame.reset_index(drop=True)
        self.root, self.transform = Path(image_root), transform
        self.with_labels = with_labels

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.file_name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row.label) if self.with_labels else -1


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def image_loader(frame, root, cfg, batch, training=False, flip=False, seed=0, with_labels=True):
    return DataLoader(ImageDataset(frame, root, image_transform(cfg, training, flip), with_labels),
                      batch_size=batch, shuffle=training, drop_last=False,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=worker_seed)


def grid_features(activation):
    return torch.cat([F.adaptive_avg_pool2d(activation, (2, 4)).flatten(1),
                      F.adaptive_avg_pool2d(activation, 1).flatten(1)], dim=1)


class DinoClassifier(nn.Module):
    def __init__(self, spec, pretrained, weights_dir, drop_path, checkpointing):
        super().__init__()
        import timm
        self.backbone = timm.create_model(spec.architecture, pretrained=False, num_classes=0,
                                          dynamic_img_size=True, drop_path_rate=drop_path)
        if pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            path = hf_hub_download("timm/" + spec.architecture, "model.safetensors",
                                   revision=spec.revision, cache_dir=str(weights_dir))
            if file_sha256(path) != spec.weights_sha256:
                raise ValueError("Pretrained DINO checksum mismatch")
            state = {k: v for k, v in load_file(path).items() if not k.startswith("head.")}
            self.backbone.load_state_dict(state, strict=True)
        self.backbone.set_grad_checkpointing(checkpointing)
        dim = self.backbone.embed_dim * 10
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, 5)

    def forward(self, images):
        tokens = self.backbone.forward_features(images)
        patch_h, patch_w = self.backbone.patch_embed.patch_size
        h, w = images.shape[2] // patch_h, images.shape[3] // patch_w
        patches = tokens[:, self.backbone.num_prefix_tokens:]
        if patches.shape[1] != h * w:
            raise ValueError("Patch grid/prefix token mismatch")
        activation = patches.transpose(1, 2).reshape(len(images), -1, h, w)
        features = torch.cat([tokens[:, 0], grid_features(activation)], dim=1)
        return self.head(self.norm(features.float()))


class SpatialClassifier(nn.Module):
    def __init__(self, name, pretrained, checkpointing):
        super().__init__()
        self.name = name
        if name == "resnet18":
            network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone, channels = nn.Sequential(*list(network.children())[:-2]), 512
        elif name == "convnext_tiny":
            network = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone, channels = network.features, 768
        elif name == "swin_t":
            network = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone, self.final_norm, channels = network.features, network.norm, 768
        else:
            raise ValueError(name)
        self.checkpointing = checkpointing and name != "resnet18"
        self.norm, self.head = nn.LayerNorm(channels * 9), nn.Linear(channels * 9, 5)

    def forward(self, images):
        if self.training and self.checkpointing:
            activation = checkpoint(self.backbone, images, use_reentrant=False)
        else:
            activation = self.backbone(images)
        if self.name == "swin_t":
            activation = self.final_norm(activation).permute(0, 3, 1, 2)
        return self.head(self.norm(grid_features(activation).float()))


def build_model(name, cfg, pretrained=True, weights_dir=None):
    if name not in MODEL_SPECS:
        raise ValueError(name)
    if name == "dinov2_large" and not cfg.allow_large:
        raise ValueError("Enable allow_large before creating DINOv2-L")
    spec = MODEL_SPECS[name]
    if spec.family == "dino":
        return DinoClassifier(spec, pretrained, weights_dir, cfg.drop_path,
                              cfg.gradient_checkpointing)
    return SpatialClassifier(name, pretrained, cfg.gradient_checkpointing)


def optimizer_groups(model, spec, cfg):
    """Apply backbone LLRD and exclude biases/norms from weight decay."""
    groups = {}
    depth = len(model.backbone.blocks) if spec.family == "dino" else 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith(("backbone.", "final_norm."))
        multiplier = 1.0
        if backbone and spec.family == "dino":
            if name.startswith("backbone.blocks."):
                layer = int(name.split(".")[2]) + 1
            elif name.startswith(("backbone.norm.", "backbone.fc_norm.")):
                layer = depth + 1
            else:
                layer = 0
            multiplier = spec.llrd ** (depth + 1 - layer)
        lr = cfg.backbone_lr * multiplier if backbone else cfg.head_lr
        decay = cfg.weight_decay if parameter.ndim > 1 else 0.0
        key = (lr, decay, backbone)
        groups.setdefault(key, []).append(parameter)
    return [dict(params=params, lr=lr, weight_decay=decay, backbone=bb)
            for (lr, decay, bb), params in groups.items()]


@torch.inference_mode()
def predict(model, frame, image_root, cfg, batch, device, flip=False, with_labels=True):
    model.eval()
    loader = image_loader(frame, image_root, cfg, batch, flip=flip, with_labels=with_labels)
    outputs = []
    for images, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        outputs.append(logits.float().softmax(1).cpu().numpy())
    probabilities = np.concatenate(outputs)
    validate_probabilities(probabilities, len(frame))
    return probabilities


def validate_probabilities(probabilities, rows):
    if probabilities.shape != (rows, 5) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability dimensions/values")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probability distribution")


def metrics(probabilities, labels):
    validate_probabilities(probabilities, len(labels))
    return {"macro_f1": float(f1_score(labels, probabilities.argmax(1), labels=list(range(5)),
                                        average="macro", zero_division=0)),
            "val_loss": float(log_loss(labels, probabilities, labels=list(range(5))))}


def train_member(name, frame, data, cfg, directory, device, scope):
    """Fixed epoch budget, no outer-label early stopping. On OOM restart with a smaller micro batch."""
    spec = MODEL_SPECS[name]
    batch = cfg.batch_overrides.get(name, spec.micro_batch)
    directory.mkdir(parents=True, exist_ok=False)
    while True:
        try:
            return _train_member(name, frame, data, cfg, directory, device, scope, batch)
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                write_json(directory / "failure.json", {"reason": "CUDA OOM at micro_batch=1", "model": name})
                raise
        # Outside except: release the traceback and its model/optimizer tensors first.
        gc.collect()
        torch.cuda.empty_cache()
        batch //= 2
        with (directory / "oom_retries.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"model": name, "restart_micro_batch": batch}) + "\n")
        print(f"{name}: OOM; restart from pretrained weights with micro_batch={batch}", flush=True)


def _train_member(name, frame, data, cfg, directory, device, scope, batch):
    spec = MODEL_SPECS[name]
    seed_everything(spec.seed)
    model = build_model(name, cfg, True, directory.parent.parent / "weights").to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, spec, cfg), betas=(.9, .999))
    accumulation = cfg.effective_batch // batch
    steps_per_epoch = math.ceil(math.ceil(len(frame) / batch) / accumulation)
    total_steps = steps_per_epoch * cfg.epochs
    warm_steps = max(1, steps_per_epoch * cfg.warmup_epochs)

    def factor(step, backbone):
        cosine = .1 + .9 * .5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
        return cosine * (min(1., (step + 1) / warm_steps) if backbone and cfg.warmup_epochs else 1.)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lambda step, bb=g["backbone"]: factor(step, bb) for g in optimizer.param_groups])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=device.type == "cuda")
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        loader = image_loader(frame, data.data_dir / "train", cfg, batch, training=True,
                              seed=spec.seed * 100 + epoch + 1)
        optimizer.zero_grad(set_to_none=True)
        started, total_loss, seen = time.perf_counter(), 0., 0
        for step, (images, labels) in enumerate(loader):
            start = (step // accumulation) * accumulation
            end = min(start + accumulation, len(loader))
            group_samples = min(len(frame) - start * batch, (end - start) * batch)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            loss_sum = F.cross_entropy(logits.float(), labels.to(device), reduction="sum")
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss_sum / group_samples).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss_sum.detach())
            seen += len(labels)
        assert seen == len(frame)
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch + 1, "train_loss": total_loss / seen, "rows": seen,
               "seconds": elapsed, "micro_batch": batch, "effective_batch": cfg.effective_batch}
        history.append(row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(name, scope, row, flush=True)
        if epoch == 0 and elapsed * cfg.epochs > cfg.maximum_hours_per_model * 3600:
            raise RuntimeError("Time guard: projected training exceeds configured per-model budget")
    path = directory / "final.pt"
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "model_name": name, "spec": asdict(spec), "config": asdict(cfg),
                "class_names": list(CLASS_NAMES), "data_identity": data.identity,
                "scope": scope, "epochs": cfg.epochs, "micro_batch": batch}, path)
    del optimizer, scheduler, scaler
    return model.eval(), path, batch


def load_model(path, data, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["class_names"] != list(CLASS_NAMES) or saved["data_identity"] != data.identity:
        raise ValueError("Checkpoint class mapping/data identity mismatch")
    cfg = PipelineConfig(**saved["config"])
    name = saved["model_name"]
    if saved["spec"] != asdict(MODEL_SPECS[name]):
        raise ValueError("Model spec has changed since checkpoint creation")
    model = build_model(name, cfg, pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), cfg, saved["micro_batch"]


def choose_inner_recipe(predictions, labels, cfg):
    """Only inner selection labels may influence views/members/weights."""
    choices, candidates = {}, []
    for name, views in predictions.items():
        plain, flip = views["plain"], (views["plain"] + views["flip"]) / 2
        use_flip = metrics(flip, labels)["macro_f1"] >= metrics(plain, labels)["macro_f1"] + cfg.flip_gain
        choices[name] = {"tta": bool(use_flip), "probabilities": flip if use_flip else plain}
    ranking = sorted(choices, key=lambda name: metrics(choices[name]["probabilities"], labels)["macro_f1"], reverse=True)
    member_sets = [(name,) for name in ranking]
    member_sets += [tuple(ranking[:n]) for n in range(2, len(ranking) + 1)]
    dino_members = tuple(name for name in ranking if name.startswith("dinov2_"))
    if len(dino_members) > 1:
        member_sets.append(dino_members)
    for members in dict.fromkeys(member_sets):
        probability = np.mean([choices[name]["probabilities"] for name in members], axis=0)
        candidates.append({"members": [{"name": name, "weight": 1 / len(members), "tta": choices[name]["tta"]}
                                        for name in members], **metrics(probability, labels)})
    selected = max(candidates, key=lambda c: (c["macro_f1"], -c["val_loss"], -len(c["members"])))
    return selected, candidates


def blend_members(predictions, members):
    if not members or not np.isclose(sum(m["weight"] for m in members), 1):
        raise ValueError("Invalid ensemble weights")
    combined = None
    for member in members:
        views = predictions[member["name"]]
        probability = (views["plain"] + views["flip"]) / 2 if member["tta"] else views["plain"]
        combined = member["weight"] * probability if combined is None else combined + member["weight"] * probability
    validate_probabilities(combined, len(combined))
    return combined


def _macro_from_confusion(matrix):
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * np.diag(matrix), denominator,
                           out=np.zeros(5, dtype=float), where=denominator > 0).mean())


def paired_group_bootstrap(reference, candidate, labels, groups, samples=2000):
    """Resample duplicate groups together; both methods receive identical draws."""
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    count = int(inverse.max()) + 1
    matrices = np.zeros((count, 2, 5, 5), dtype=np.int32)
    for method, probabilities in enumerate((reference, candidate)):
        np.add.at(matrices[:, method], (inverse, labels, probabilities.argmax(1)), 1)
    rng, gains = np.random.default_rng(0), []
    for _ in range(samples):
        draw = rng.integers(0, count, size=count)
        summed = matrices[draw].sum(0)
        gains.append(_macro_from_confusion(summed[1]) - _macro_from_confusion(summed[0]))
    gains = np.asarray(gains)
    return {"p_gain_nonpositive": float((gains <= 0).mean()),
            "gain_ci95": np.quantile(gains, [.025, .975]).tolist(),
            "samples": samples, "unit": "duplicate_group",
            "note": "Descriptive resampling on reused development outer; not independent test evidence."}


def external_reference(reference_dir, data):
    reference_dir = Path(reference_dir)
    info = json.loads((reference_dir / "run_info.json").read_text(encoding="utf-8"))
    if info["split_metadata"]["split_manifest_sha256"] != data.identity["split_manifest_sha256"]:
        raise ValueError("External baseline was evaluated on another split")
    if info["class_names"] != list(CLASS_NAMES):
        raise ValueError("External baseline class mapping mismatch")
    with np.load(reference_dir / "best_validation.npz", allow_pickle=False) as saved:
        ids, p, targets = saved["image_id"], saved["probabilities"], saved["targets"]
    if not pd.Index(ids).is_unique or len(ids) != len(data.outer):
        raise ValueError("Invalid external baseline IDs")
    order = pd.Index(ids).get_indexer(data.outer.id)
    if (order < 0).any() or not np.array_equal(targets[order], data.outer.label.astype(int)):
        raise ValueError("External baseline IDs/labels mismatch")
    validate_probabilities(p[order], len(data.outer))
    return p[order], {"source": str(reference_dir), "sha256": file_sha256(reference_dir / "best_validation.npz")}


def assert_run_identity(run_dir, data):
    info = json.loads((Path(run_dir) / "experiment.json").read_text(encoding="utf-8"))
    if info["data_identity"] != data.identity:
        raise ValueError("Run belongs to another dataset/split")
    return info


def run_development(data, cfg, output_root, reference_dir=None, device=None):
    cfg.validate()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(output_root) / ("dinov2_ensemble_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "experiment.json", {
        "config": asdict(cfg), "data_identity": data.identity,
        "code_sha256": file_sha256(__file__), "model_specs": {name: asdict(MODEL_SPECS[name]) for name in cfg.model_names},
        "software": {package: version(package) for package in ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn")},
        "device": str(device), "reference_dir": str(reference_dir) if reference_dir else None,
        "status": "DEVELOPMENT", "final_refit_validation_score": None,
    })
    pd.concat([data.fit.assign(partition="fit"), data.selection.assign(partition="selection"),
               data.outer.assign(partition="outer")]).to_csv(run_dir / "partitions.csv", index=False)
    selection_predictions, outer_predictions, checkpoints = {}, {}, {}
    try:
        for name in cfg.model_names:
            model, path, batch = train_member(name, data.fit, data, cfg, run_dir / "development" / name,
                                              device, "inner_fit_fixed_epochs")
            checkpoints[name] = str(path)
            selection_predictions[name], outer_predictions[name] = {}, {}
            for view in ("plain", "flip"):
                selection_predictions[name][view] = predict(model, data.selection, data.data_dir / "train", cfg, batch, device, view == "flip")
                outer_predictions[name][view] = predict(model, data.outer, data.data_dir / "train", cfg, batch, device, view == "flip")
            np.savez_compressed(path.parent / "validation.npz", **{
                "selection_ids": data.selection.id.to_numpy(dtype=str), "outer_ids": data.outer.id.to_numpy(dtype=str),
                **{f"selection_{v}": p for v, p in selection_predictions[name].items()},
                **{f"outer_{v}": p for v, p in outer_predictions[name].items()}})
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        selected, candidates = choose_inner_recipe(selection_predictions, data.selection.label.astype(int).to_numpy(), cfg)
        combined = blend_members(outer_predictions, selected["members"])
        if reference_dir:
            reference, reference_info = external_reference(reference_dir, data)
        else:
            reference = outer_predictions["resnet18"]["plain"]
            reference_info = {"source": "new resnet18 plain, same inner fit partition"}
        truth = data.outer.label.astype(int).to_numpy()
        reference_metrics, candidate_metrics = metrics(reference, truth), metrics(combined, truth)
        bootstrap = paired_group_bootstrap(reference, combined, truth, data.outer.duplicate_group, cfg.bootstrap_samples)
        gain = candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
        passed = gain >= cfg.minimum_gain and bootstrap["p_gain_nonpositive"] < cfg.bootstrap_pmax
        recipe = {"selected_on": "inner selection only", "selected": selected, "inner_candidates": candidates,
                  "development_checkpoints": checkpoints, "reference": reference_info,
                  "outer_reference": reference_metrics, "outer_candidate": candidate_metrics,
                  "outer_gain": gain, "bootstrap": bootstrap, "passes": bool(passed),
                  "data_identity": data.identity, "outer_note": data.identity["outer_note"]}
        write_json(run_dir / "selection.json", recipe)
        np.savez_compressed(run_dir / "outer_validation.npz", probabilities=combined,
                            reference_probabilities=reference, targets=truth,
                            image_id=data.outer.id.to_numpy(dtype=str))
        write_json(run_dir / "status.json", {"phase": "SELECTED" if passed else "NO_CANDIDATE", "outer_gain": gain})
        return run_dir, recipe
    except Exception as error:
        write_json(run_dir / "status.json", {"phase": "FAILED", "error": str(error), "type": type(error).__name__})
        raise


def refit_selected(run_dir, data, device=None):
    run_dir = Path(run_dir)
    info = assert_run_identity(run_dir, data)
    recipe = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    if not recipe["passes"]:
        raise ValueError("Candidate failed the fixed outer gate; do not refit/export it")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg, paths = PipelineConfig(**info["config"]), {}
    for member in recipe["selected"]["members"]:
        name = member["name"]
        model, path, _ = train_member(name, data.train, data, cfg, run_dir / "full_refit" / name,
                                      device, "all_train_fixed_epochs")
        paths[name] = str(path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(run_dir / "refit.json", {"checkpoints": paths, "validation_score_of_refit": None,
                                       "epochs": cfg.epochs, "rows": len(data.train)})
    return paths


def export_test(run_dir, data, sample_path, submission_root, use_refit=False, device=None):
    run_dir = Path(run_dir)
    assert_run_identity(run_dir, data)
    recipe = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    if not recipe["passes"]:
        raise ValueError("No candidate: outer gate did not pass")
    # Re-check raw files before any test use; never consume test labels/statistics for selection.
    if file_sha256(data.data_dir / "test.csv") != data.identity["test_csv_sha256"]:
        raise ValueError("Test CSV changed since initialization")
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    if not sample.id.is_unique or sample.id.eq("").any() or len(sample) != len(data.test):
        raise ValueError("Invalid sample IDs")
    order = pd.Index(data.test.id).get_indexer(sample.id)
    if (order < 0).any():
        raise ValueError("Sample/test IDs do not match")
    paths = recipe["development_checkpoints"]
    if use_refit:
        paths = json.loads((run_dir / "refit.json").read_text(encoding="utf-8"))["checkpoints"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = np.zeros((len(data.test), 5), dtype=np.float64)
    checkpoint_hashes = {}
    for member in recipe["selected"]["members"]:
        name, path = member["name"], Path(paths[member["name"]])
        model, cfg, batch = load_model(path, data, device)
        probability = predict(model, data.test, data.data_dir / "test", cfg, batch, device, with_labels=False)
        if member["tta"]:
            probability = (probability + predict(model, data.test, data.data_dir / "test", cfg, batch,
                                                  device, flip=True, with_labels=False)) / 2
        result += member["weight"] * probability
        checkpoint_hashes[name] = file_sha256(path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    validate_probabilities(result, len(data.test))
    result = result[order].astype(np.float32)
    output = pd.DataFrame({"id": sample.id, "task4": result.argmax(1).astype(np.int64)})
    destination = Path(submission_root) / (run_dir.name + ("_refit" if use_refit else "_development"))
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / "task4.csv"
    output.to_csv(csv_path, index=False, lineterminator="\n")
    pd.testing.assert_frame_equal(pd.read_csv(csv_path, dtype={"id": str}), output)
    np.savez_compressed(destination / "test_probabilities.npz", probabilities=result,
                        image_id=sample.id.to_numpy(dtype=str))
    write_json(destination / "prediction_metadata.json", {
        "members": recipe["selected"]["members"], "checkpoint_sha256": checkpoint_hashes,
        "csv_sha256": file_sha256(csv_path), "data_identity": data.identity, "rows": len(output),
        "full_train_refit": use_refit, "development_outer_metrics": recipe["outer_candidate"],
        "validation_score_of_refit": None, "kaggle_score": None,
    })
    return csv_path
