import hashlib
import json
import random
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from datadings.sets.ImageNet21k_synsets import (
    NUM_TRAIN_SAMPLES,
    NUM_VAL_SAMPLES,
    NUM_VALID_SYNSETS,
    SYNSET_LIST,
    SYNSET_WORDS,
)
from src.dataset._loader import ImageNetLoader

# CONFIGURATION

IMAGENET21K_P_ROOT = Path("./data/imagenet21k_p")

IMAGENET_W21_SYNSET_URL = (
    "https://image-net.org/data/winter21_whole/{wnid}.tar"
)

USER_AGENT = "Mozilla/5.0 (compatible; ImageNet-EDA-Research/1.0)"

NUM_CLASSES = 10_450
VALIDATION_IMAGES_PER_CLASS = 50
SAMPLE_CLASSES = 20
SAMPLE_TRAIN_IMAGES_PER_CLASS = 100

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

train_transform = transforms.Compose([
])
test_transform = transforms.Compose([
])


class ImageNet21KPMetadataManager:
    """Handles the exact 10,450 ImageNet-21K-P Winter21 classes."""

    def __init__(self, root=IMAGENET21K_P_ROOT):
        self.root = Path(root)
        self.metadata_root = self.root / "metadata"
        self.metadata_root.mkdir(parents=True, exist_ok=True)

        self.synset_cache = self.metadata_root / "imagenet21k_p_synsets.txt"
        self.names_cache = self.metadata_root / "synset_names.json"

        if NUM_VALID_SYNSETS != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES:,} ImageNet-21K-P classes, "
                f"but metadata contains {NUM_VALID_SYNSETS:,}."
            )

        self.synsets = list(SYNSET_LIST)
        self.synset_to_name = dict(SYNSET_WORDS)

        self._save_metadata()

    def _save_metadata(self):
        if not self.synset_cache.exists():
            self.synset_cache.write_text(
                "\n".join(self.synsets) + "\n",
                encoding="utf-8",
            )

        if not self.names_cache.exists():
            names = {
                wnid: self.synset_to_name.get(wnid, wnid)
                for wnid in self.synsets
            }

            self.names_cache.write_text(
                json.dumps(names, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def get_synset_ids(self):
        return self.synsets.copy()

    def get_synset_names(self, synsets=None):
        synsets = self.synsets if synsets is None else synsets

        return {
            wnid: self.synset_to_name.get(wnid, wnid)
            for wnid in synsets
        }


class ImageNet21KPDownloader:
    def __init__(
            self,
            metadata_manager,
            mode="sample",
            root=IMAGENET21K_P_ROOT,
            sample_classes=SAMPLE_CLASSES,
            train_images_per_class=SAMPLE_TRAIN_IMAGES_PER_CLASS,
            validation_images_per_class=VALIDATION_IMAGES_PER_CLASS,
            seed=42,
    ):
        if mode not in {"sample", "full"}:
            raise ValueError("mode must be 'sample' or 'full'")

        if validation_images_per_class <= 0:
            raise ValueError("validation_images_per_class must be positive.")

        if train_images_per_class is not None and train_images_per_class <= 0:
            raise ValueError("train_images_per_class must be positive or None.")

        self.metadata_manager = metadata_manager
        self.mode = mode
        self.root = Path(root)
        self.sample_classes = sample_classes
        self.train_images_per_class = train_images_per_class
        self.validation_images_per_class = validation_images_per_class
        self.seed = seed

        self.images_root = self.root / "images"
        self.train_root = self.images_root / "train"
        self.validation_root = self.images_root / "validation"

    def download(self):
        candidates = self.metadata_manager.get_synset_ids()

        self.train_root.mkdir(parents=True, exist_ok=True)
        self.validation_root.mkdir(parents=True, exist_ok=True)

        if self.mode == "sample":
            classes = self._select_sample_classes(candidates)
            self._print_sample_summary(classes)
        else:
            classes = candidates
            self._print_full_summary(classes)

        progress = tqdm(
            classes,
            desc=f"Building ImageNet-21K-P {self.mode}",
            unit="class",
            dynamic_ncols=True,
        )

        processed = []

        for wnid in progress:
            progress.set_postfix(wnid=wnid)

            if self._prepare_class(wnid):
                processed.append(wnid)

        if len(processed) != len(classes):
            raise RuntimeError(
                f"Expected {len(classes):,} classes, "
                f"but only {len(processed):,} were prepared."
            )

        return processed, self.train_root, self.validation_root, "imagenet21k_p"

    def _select_sample_classes(self, candidates):
        if self.sample_classes > len(candidates):
            raise ValueError(
                f"sample_classes={self.sample_classes:,} exceeds "
                f"{len(candidates):,} available P classes."
            )

        rng = random.Random(self.seed)
        return sorted(rng.sample(list(candidates), self.sample_classes))

    def _print_sample_summary(self, classes):
        print("\nImageNet-21K-P Winter21 Sample")
        print("-" * 55)
        print(f"Classes               : {len(classes):,}")

        if self.train_images_per_class is None:
            print("Train images/class    : ALL")
            print("Expected train        : depends on selected classes")
        else:
            print(f"Train images/class    : {self.train_images_per_class:,}")
            print(
                f"Expected train        : "
                f"{len(classes) * self.train_images_per_class:,}"
            )

        print(f"Validation/class      : {self.validation_images_per_class:,}")
        print(
            f"Expected validation   : "
            f"{len(classes) * self.validation_images_per_class:,}"
        )
        print(f"Output folder         : {self.images_root}")
        print("-" * 55)

    def _print_full_summary(self, classes):
        print("\nImageNet-21K-P Winter21 Full")
        print("-" * 55)
        print(f"Classes               : {len(classes):,}")
        print(f"Expected train        : {NUM_TRAIN_SAMPLES:,}")
        print(f"Expected validation   : {NUM_VAL_SAMPLES:,}")
        print(f"Validation/class      : {self.validation_images_per_class:,}")
        print(f"Output folder         : {self.images_root}")
        print("-" * 55)

    # ========================================================
    # PREPARE ONE CLASS
    # ========================================================

    def _prepare_class(self, wnid):
        train_dir = self.train_root / wnid
        validation_dir = self.validation_root / wnid
        complete_file = train_dir / ".complete"

        train_count = self._count_images(train_dir)
        validation_count = self._count_images(validation_dir)

        if complete_file.exists():
            if validation_count >= self.validation_images_per_class:
                if self.mode == "full" or self.train_images_per_class is None:
                    return train_count > 0

                if train_count >= self.train_images_per_class:
                    return True

        if (
                self.mode == "sample"
                and self.train_images_per_class is not None
                and train_count >= self.train_images_per_class
                and validation_count >= self.validation_images_per_class
        ):
            return True

        return self._download_and_process_tar(
            wnid,
            train_dir,
            validation_dir,
            complete_file,
        )

    def _download_and_process_tar(self, wnid, train_dir, validation_dir, complete_file):
        url = IMAGENET_W21_SYNSET_URL.format(wnid=wnid)

        with tempfile.TemporaryDirectory(prefix=f"imagenet21kp_{wnid}_") as temp_dir:
            tar_path = Path(temp_dir) / f"{wnid}.tar"

            try:
                self._download_file(url, tar_path)

            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return False

                if error.code in {401, 403}:
                    raise RuntimeError(
                        "ImageNet denied access to this synset TAR. "
                        "Make sure ImageNet access is valid and its terms "
                        f"have been accepted. URL: {url}"
                    ) from error

                raise

            return self._process_tar(
                wnid,
                tar_path,
                train_dir,
                validation_dir,
                complete_file,
            )

    @staticmethod
    def _download_file(url, destination, timeout=180, chunk_size=8 * 1024 * 1024):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(destination, "wb") as output:
                while True:
                    chunk = response.read(chunk_size)

                    if not chunk:
                        break

                    output.write(chunk)

        return destination

    def _process_tar(self, wnid, tar_path, train_dir, validation_dir, complete_file):
        with tarfile.open(tar_path, "r:*") as tar:
            members = sorted(
                (
                    member
                    for member in tar.getmembers()
                    if member.isfile()
                       and Path(member.name).suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda member: Path(member.name).name,
            )

            if len(members) <= self.validation_images_per_class:
                return False

            validation_members = members[:self.validation_images_per_class]
            train_candidates = members[self.validation_images_per_class:]
            train_members = self._select_train_items(wnid, train_candidates)

            if not train_members:
                return False

            self._process_tar_members(
                tar,
                validation_members,
                validation_dir,
            )

            self._process_tar_members(
                tar,
                train_members,
                train_dir,
            )

        if self.mode == "full" or self.train_images_per_class is None:
            complete_file.touch()

        return (
                self._count_images(validation_dir) >= self.validation_images_per_class
                and self._train_output_complete(train_dir, len(train_candidates))
        )

    def _select_train_items(self, wnid, items):
        items = list(items)

        if self.mode == "full" or self.train_images_per_class is None:
            return items

        if len(items) < self.train_images_per_class:
            return []

        rng = random.Random(f"{self.seed}:{wnid}")
        return rng.sample(items, self.train_images_per_class)

    def _train_output_complete(self, train_dir, full_train_count):
        train_count = self._count_images(train_dir)

        if self.mode == "full" or self.train_images_per_class is None:
            return train_count >= full_train_count

        return train_count >= self.train_images_per_class

    # IMAGE PREPROCESSING to .JPEG
    def _process_tar_members(self, tar, members, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

        for member in members:
            filename = Path(member.name).stem + ".JPEG"
            output_path = output_dir / filename

            if output_path.exists():
                continue

            source = tar.extractfile(member)

            if source is None:
                continue

            with source, Image.open(source) as image:
                self._save_processed_image(image, output_path)

    def _save_processed_image(self, image, output_path):
        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)

    @staticmethod
    def _count_images(folder):
        folder = Path(folder)

        if not folder.exists():
            return 0

        return sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in folder.iterdir()
        )


class ImageNet21KPDataset(Dataset):
    """One local dataset class for train, validation, test and subsets."""

    def __init__(
            self,
            root=None,
            classes=None,
            class_to_name=None,
            transform=None,
            base_dataset=None,
            indices=None,
    ):
        self.transform = transform

        if base_dataset is not None:
            if indices is None:
                raise ValueError("indices are required when base_dataset is provided.")

            self.base_dataset = base_dataset
            self.indices = [int(index) for index in indices]

            self.root = base_dataset.root
            self.classes = base_dataset.classes
            self.class_to_idx = base_dataset.class_to_idx
            self.class_to_name = base_dataset.class_to_name
            self.dataset_metadata = base_dataset.dataset_metadata
            self.targets = [base_dataset.targets[index] for index in self.indices]
            return

        if root is None or classes is None:
            raise ValueError("root and classes are required.")

        self.base_dataset = None
        self.indices = None
        self.root = Path(root)
        self.classes = list(classes)

        class_to_name = class_to_name or {}

        self.class_to_idx = {
            wnid: index
            for index, wnid in enumerate(self.classes)
        }

        self.class_to_name = {
            wnid: class_to_name.get(wnid, wnid)
            for wnid in self.classes
        }

        self.dataset_metadata = {
            "dataset": "ImageNet-21K-P Winter21",
            "num_classes": len(self.classes),
            "source_num_classes": NUM_CLASSES,
        }

        self.samples = []

        for wnid in self.classes:
            folder = self.root / wnid

            if not folder.exists():
                continue

            target = self.class_to_idx[wnid]

            self.samples.extend(
                (path, target)
                for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

        self.targets = [target for _, target in self.samples]

    def subset(self, indices, transform=None):
        return ImageNet21KPDataset(
            base_dataset=self,
            indices=indices,
            transform=transform,
        )

    def __len__(self):
        return len(self.indices) if self.base_dataset is not None else len(self.samples)

    def __getitem__(self, index):
        if self.base_dataset is not None:
            base_index = self.indices[index]
            path, target = self.base_dataset.samples[base_index]
        else:
            path, target = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image, target


# VALIDATION / TEST SPLITTER
class DatasetSplitter:
    def __init__(self, val_ratio=0.5, test_ratio=0.5, seed=42):
        self.val_ratio, self.test_ratio = self._normalize_ratios(val_ratio, test_ratio)
        self.seed = seed

    @staticmethod
    def _normalize_ratios(val_ratio, test_ratio):
        ratios = [float(val_ratio), float(test_ratio)]

        if any(ratio < 0 for ratio in ratios):
            raise ValueError("Split ratios cannot be negative.")

        total = sum(ratios)

        if total <= 0:
            raise ValueError("At least one split ratio must be positive.")

        return tuple(ratio / total for ratio in ratios)

    def split(self, dataset):
        rng = random.Random(self.seed)
        class_indices = {}

        for index, target in enumerate(dataset.targets):
            class_indices.setdefault(int(target), []).append(index)

        val_indices = []
        test_indices = []

        for target in sorted(class_indices):
            indices = class_indices[target].copy()
            rng.shuffle(indices)

            n_val = int(round(len(indices) * self.val_ratio))
            n_val = min(max(n_val, 0), len(indices))

            val_indices.extend(indices[:n_val])
            test_indices.extend(indices[n_val:])

        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        return val_indices, test_indices

    def get_or_create_split(self, dataset, cache_root):
        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)

        signature = self._dataset_signature(dataset)

        split_file = cache_root / (
            f"validation_split_{self.val_ratio:.6f}_{self.test_ratio:.6f}_"
            f"seed{self.seed}_{signature}.json"
        )

        if split_file.exists():
            try:
                split = json.loads(split_file.read_text(encoding="utf-8"))

                if (
                        split.get("dataset_size") == len(dataset)
                        and split.get("num_classes") == len(dataset.classes)
                        and split.get("signature") == signature
                        and split.get("val_ratio") == self.val_ratio
                        and split.get("test_ratio") == self.test_ratio
                        and split.get("seed") == self.seed
                ):
                    return split["val"], split["test"]

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

            split_file.unlink()

        val_indices, test_indices = self.split(dataset)

        split = {
            "dataset_size": len(dataset),
            "num_classes": len(dataset.classes),
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "signature": signature,
            "val": [int(index) for index in val_indices],
            "test": [int(index) for index in test_indices],
        }

        split_file.write_text(
            json.dumps(split, indent=2),
            encoding="utf-8",
        )

        return split["val"], split["test"]

    @staticmethod
    def _dataset_signature(dataset):
        value = "|".join(dataset.classes) + f"|{len(dataset)}"
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


class ImageNet21KPLoader(ImageNetLoader):
    def __init__(
            self,
            mode="sample",
            root=IMAGENET21K_P_ROOT,
            sample_classes=SAMPLE_CLASSES,
            train_images_per_class=SAMPLE_TRAIN_IMAGES_PER_CLASS,
            validation_images_per_class=VALIDATION_IMAGES_PER_CLASS,
            val_ratio=0.5,
            test_ratio=0.5,
            seed=42,
    ):
        if mode not in {"sample", "full"}:
            raise ValueError("mode must be 'sample' or 'full'")

        self.mode = mode
        self.root = Path(root)
        self.sample_classes = sample_classes
        self.train_images_per_class = train_images_per_class
        self.validation_images_per_class = validation_images_per_class
        self.seed = seed

        self.metadata_manager = ImageNet21KPMetadataManager(self.root)

        self.splitter = DatasetSplitter(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    def load(self):
        downloader = ImageNet21KPDownloader(
            metadata_manager=self.metadata_manager,
            mode=self.mode,
            root=self.root,
            sample_classes=self.sample_classes,
            train_images_per_class=self.train_images_per_class,
            validation_images_per_class=self.validation_images_per_class,
            seed=self.seed,
        )

        classes, train_root, validation_root, dataset_name = downloader.download()
        class_to_name = self.metadata_manager.get_synset_names(classes)

        train_dataset = ImageNet21KPDataset(
            root=train_root,
            classes=classes,
            class_to_name=class_to_name,
            transform=train_transform,
        )

        official_validation = ImageNet21KPDataset(
            root=validation_root,
            classes=classes,
            class_to_name=class_to_name,
        )

        val_indices, test_indices = self.splitter.get_or_create_split(official_validation, self.root / "splits")
        val_dataset = official_validation.subset(val_indices, transform=test_transform)
        test_dataset = official_validation.subset(test_indices, transform=test_transform)
        return (train_dataset, val_dataset, test_dataset, dataset_name)


# imagenet_21k_p = ImageNet21KPLoader(
#     mode="sample",
#     sample_classes=200,
#     train_images_per_class=None,
#     seed=42,
# ).load()

# cleaned_imagenet_21k_p = model_ready_dataset.ModelReadyDatasetLoader(
#     ImageNet21KPLoader(
#         mode="sample",
#         sample_classes=200,
#         train_images_per_class=None,
#         seed=42,
#     )
# ).load()
