# import hashlib
# import json
# import random
# import re
# import tarfile
# import tempfile
# import urllib.error
# import urllib.request
# from collections import defaultdict
# from pathlib import Path
#
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# from tqdm.auto import tqdm
# from src.dataset._loader import ImageNetLoader
#
# # CONFIGURATION
#
# FULL_ROOT = Path("./data/imagenet21k")
#
# IMAGENET_W21_SYNSET_URL = (
#     "https://image-net.org/data/winter21_whole/{wnid}.tar"
# )
#
# FALLBACK_SYNSET_IDS_URL = (
#     "https://raw.githubusercontent.com/ailia-ai/ailia-models/"
#     "master/object_detection/detic/datasets_detic/"
#     "imagenet21k_wordnet_ids.txt"
# )
#
# SYNSET_LIST_URLS = [
#     "https://www.image-net.org/api/text/imagenet.synset.obtain_synset_list",
#     FALLBACK_SYNSET_IDS_URL,
# ]
#
# FALLBACK_LABELS_URL = (
#     "https://raw.githubusercontent.com/vkinakh/"
#     "stable-diffusion-imagenet/main/"
#     "imagenet21k_wordnet_lemmas.txt"
# )
#
# OFFICIAL_WORDS_URL = "https://image-net.org/archive/words.txt"
#
# USER_AGENT = "Mozilla/5.0 (compatible; ImageNet-EDA-Research/1.0)"
#
# IMAGE_EXTENSIONS = {
#     ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
# }
#
# SAMPLE_NUM_CLASSES = 200
# SAMPLE_IMAGES_PER_CLASS = 100
#
# train_transform = transforms.Compose([
# ])
# test_transform = transforms.Compose([
# ])
#
#
# class ImageNetMetadataManager:
#     """Handles ImageNet synset IDs and human-readable class names."""
#
#     def __init__(self, root=FULL_ROOT):
#         self.root = Path(root)
#         self.metadata_root = self.root / "metadata"
#         self.metadata_root.mkdir(parents=True, exist_ok=True)
#
#         self.synset_cache = self.metadata_root / "imagenet21k_synsets.txt"
#         self.names_cache = self.metadata_root / "synset_names.json"
#
#     @staticmethod
#     def is_valid_wnid(value):
#         return bool(re.fullmatch(r"n\d{8}", str(value).strip()))
#
#     @staticmethod
#     def _request(url, method="GET", headers=None):
#         request_headers = {"User-Agent": USER_AGENT}
#
#         if headers:
#             request_headers.update(headers)
#
#         return urllib.request.Request(
#             url,
#             method=method,
#             headers=request_headers,
#         )
#
#     @classmethod
#     def download_text(cls, url, timeout=60):
#         request = cls._request(url)
#
#         with urllib.request.urlopen(request, timeout=timeout) as response:
#             return response.read().decode("utf-8", errors="replace")
#
#     def get_synset_ids(self, force_refresh=False):
#         if self.synset_cache.exists() and not force_refresh:
#             synsets = [
#                 line.strip()
#                 for line in self.synset_cache.read_text(encoding="utf-8").splitlines()
#                 if self.is_valid_wnid(line)
#             ]
#
#             if synsets:
#                 return synsets
#
#         last_error = None
#
#         for url in SYNSET_LIST_URLS:
#             try:
#                 text = self.download_text(url)
#                 synsets = []
#
#                 for line in text.splitlines():
#                     parts = line.strip().split()
#
#                     if not parts:
#                         continue
#
#                     wnid = parts[0]
#
#                     if self.is_valid_wnid(wnid):
#                         synsets.append(wnid)
#
#                 synsets = list(dict.fromkeys(synsets))
#
#                 if synsets:
#                     self.synset_cache.write_text(
#                         "\n".join(synsets) + "\n",
#                         encoding="utf-8",
#                     )
#                     return synsets
#
#             except Exception as error:
#                 last_error = error
#
#         raise RuntimeError(
#             "Could not obtain ImageNet synset IDs. "
#             f"You may manually create {self.synset_cache} with one WNID per line."
#         ) from last_error
#
#     def get_synset_names(self, synsets=None):
#         mapping = {}
#
#         if self.names_cache.exists():
#             try:
#                 mapping = json.loads(self.names_cache.read_text(encoding="utf-8"))
#             except Exception:
#                 mapping = {}
#
#         if not mapping:
#             mapping = self._download_synset_names()
#
#         if synsets is None:
#             return mapping
#         return {wnid: mapping.get(wnid, wnid) for wnid in synsets}
#
#     def _download_synset_names(self):
#         mapping = {}
#
#         try:
#             text = self.download_text(OFFICIAL_WORDS_URL)
#
#             for line in text.splitlines():
#                 line = line.strip()
#
#                 if not line:
#                     continue
#
#                 if "\t" in line:
#                     wnid, label = line.split("\t", 1)
#                 else:
#                     parts = line.split(maxsplit=1)
#
#                     if len(parts) != 2:
#                         continue
#
#                     wnid, label = parts
#
#                 if self.is_valid_wnid(wnid):
#                     mapping[wnid] = label.strip()
#
#         except Exception:
#             pass
#
#         if not mapping:
#             try:
#                 ids_text = self.download_text(FALLBACK_SYNSET_IDS_URL)
#
#                 fallback_ids = [
#                     line.strip()
#                     for line in ids_text.splitlines()
#                     if self.is_valid_wnid(line.strip())
#                 ]
#
#                 labels_text = self.download_text(FALLBACK_LABELS_URL)
#
#                 labels = [
#                     label.strip().replace("_", " ")
#                     for label in labels_text.splitlines()
#                     if label.strip()
#                 ]
#
#                 mapping = {
#                     wnid: label
#                     for wnid, label in zip(fallback_ids, labels)
#                 }
#
#             except Exception:
#                 pass
#
#         if mapping:
#             self.names_cache.write_text(
#                 json.dumps(mapping, indent=2, ensure_ascii=False),
#                 encoding="utf-8",
#             )
#         return mapping
#
#
# class ImageNet21KDownloader:
#     def __init__(
#             self,
#             metadata_manager,
#             mode="sample",
#             root=FULL_ROOT,
#             sample_classes=SAMPLE_NUM_CLASSES,
#             images_per_class=SAMPLE_IMAGES_PER_CLASS,
#             seed=42,
#     ):
#         if mode not in {"sample", "full"}:
#             raise ValueError("mode must be 'sample' or 'full'")
#
#         self.metadata_manager = metadata_manager
#         self.mode = mode
#         self.root = Path(root)
#         self.sample_classes = sample_classes
#         self.images_per_class = images_per_class
#         self.seed = seed
#
#         self.images_root = self.root / "images"
#
#     def download(self):
#         candidates = self.metadata_manager.get_synset_ids()
#         self.images_root.mkdir(parents=True, exist_ok=True)
#
#         if self.mode == "sample":
#             return self._download_sample(candidates)
#
#         return self._download_full(candidates)
#
#     def _download_sample(self, candidates):
#         rng = random.Random(self.seed)
#         candidates = list(candidates)
#         rng.shuffle(candidates)
#
#         selected = []
#
#         print("\nFull ImageNet Winter21 Sample")
#         print("-" * 50)
#         print(f"Classes required     : {self.sample_classes:,}")
#
#         if self.images_per_class is None:
#             print("Images/class         : ALL")
#             print("Expected images      : depends on selected classes")
#         else:
#             print(f"Images/class         : {self.images_per_class:,}")
#             print(
#                 f"Expected images      : "
#                 f"{self.sample_classes * self.images_per_class:,}"
#             )
#
#         print(f"Download folder      : {self.images_root}")
#         print("-" * 50)
#
#         progress = tqdm(
#             total=self.sample_classes,
#             desc="Downloading sample classes",
#             unit="class",
#             dynamic_ncols=True,
#         )
#
#         try:
#             for wnid in candidates:
#                 if len(selected) >= self.sample_classes:
#                     break
#
#                 usable = self._download_synset(
#                     wnid,
#                     self.images_per_class,
#                 )
#
#                 if usable:
#                     selected.append(wnid)
#                     progress.update(1)
#                     progress.set_postfix(wnid=wnid)
#
#         finally:
#             progress.close()
#
#         if len(selected) < self.sample_classes:
#             raise RuntimeError(
#                 f"Only {len(selected):,} usable classes were obtained, "
#                 f"but {self.sample_classes:,} were requested."
#             )
#         return sorted(selected)
#
#     def _download_full(self, candidates):
#         downloaded = []
#
#         print("\nFull ImageNet Winter21")
#         print("-" * 50)
#         print(f"Candidate synsets    : {len(candidates):,}")
#         print("Images/class         : ALL")
#         print(f"Download folder      : {self.images_root}")
#         print("-" * 50)
#
#         progress = tqdm(
#             candidates,
#             desc="Downloading full ImageNet",
#             unit="class",
#             dynamic_ncols=True,
#         )
#
#         for wnid in progress:
#             progress.set_postfix(wnid=wnid)
#
#             if self._download_synset(wnid, None):
#                 downloaded.append(wnid)
#
#         if not downloaded:
#             raise RuntimeError("No ImageNet Winter21 classes were downloaded.")
#
#         print(f"Available classes    : {len(downloaded):,}")
#         return downloaded
#
#     def _download_synset(self, wnid, images_per_class):
#         class_directory = self.images_root / wnid
#         complete_file = class_directory / ".complete"
#         existing_count = self._count_images(class_directory)
#
#         # ALL images requested.
#         if images_per_class is None:
#             if complete_file.exists():
#                 return existing_count > 0
#
#         # Fixed number of images requested.
#         elif existing_count >= images_per_class:
#             return True
#
#         # If the full class was already downloaded, no need to redownload.
#         if complete_file.exists():
#             return existing_count > 0
#
#         url = IMAGENET_W21_SYNSET_URL.format(wnid=wnid)
#
#         with tempfile.TemporaryDirectory(prefix=f"imagenet_{wnid}_") as temp_dir:
#             tar_path = Path(temp_dir) / f"{wnid}.tar"
#
#             try:
#                 self._download_file(url, tar_path)
#
#             except urllib.error.HTTPError as error:
#                 if error.code == 404:
#                     return False
#
#                 if error.code in {401, 403}:
#                     raise RuntimeError(
#                         "ImageNet denied access to this synset TAR. "
#                         "Make sure your ImageNet access is valid and its terms "
#                         f"have been accepted. URL: {url}"
#                     ) from error
#
#                 raise
#
#             original_count, extracted_count, extracted_all = self._extract_synset(
#                 tar_path,
#                 class_directory,
#                 images_per_class,
#             )
#
#         if extracted_all:
#             complete_file.touch()
#
#         if images_per_class is not None and original_count < images_per_class:
#             return False
#         return extracted_count > 0
#
#     @staticmethod
#     def _download_file(
#             url,
#             destination,
#             timeout=180,
#             chunk_size=8 * 1024 * 1024,
#     ):
#         destination = Path(destination)
#
#         request = urllib.request.Request(
#             url,
#             headers={"User-Agent": USER_AGENT},
#         )
#
#         with urllib.request.urlopen(request, timeout=timeout) as response:
#             with open(destination, "wb") as output:
#                 while True:
#                     chunk = response.read(chunk_size)
#
#                     if not chunk:
#                         break
#
#                     output.write(chunk)
#         return destination
#
#     def _extract_synset(self, tar_path, class_directory, images_per_class):
#         class_directory = Path(class_directory)
#         class_directory.mkdir(parents=True, exist_ok=True)
#
#         with tarfile.open(tar_path, "r:*") as tar:
#             members = [
#                 member
#                 for member in tar.getmembers()
#                 if member.isfile()
#                    and Path(member.name).suffix.lower() in IMAGE_EXTENSIONS
#             ]
#
#             original_count = len(members)
#
#             if original_count == 0:
#                 raise RuntimeError(f"No images found in {tar_path.name}.")
#
#             if images_per_class is None or images_per_class >= original_count:
#                 selected_members = members
#                 extracted_all = True
#             else:
#                 rng = random.Random(f"{self.seed}:{tar_path.stem}")
#                 selected_members = rng.sample(members, images_per_class)
#                 extracted_all = False
#
#             for index, member in enumerate(selected_members):
#                 source = tar.extractfile(member)
#
#                 if source is None:
#                     continue
#
#                 filename = Path(member.name).name or f"{tar_path.stem}_{index:06d}.JPEG"
#                 output_path = class_directory / filename
#
#                 if output_path.exists():
#                     source.close()
#                     continue
#
#                 # with source, open(output_path, "wb") as output:
#                 #     shutil.copyfileobj(source, output)
#
#                 filename = Path(member.name).stem + ".JPEG"
#                 output_path = class_directory / filename
#
#                 with source, Image.open(source) as image:
#                     self._save_processed_image(image, output_path)
#
#         return (
#             original_count,
#             self._count_images(class_directory),
#             extracted_all,
#         )
#
#     def _save_processed_image(self, image, output_path):
#         image = image.convert("RGB")
#         image.save(output_path, format="JPEG", quality=95)
#
#     @staticmethod
#     def _count_images(class_directory):
#         class_directory = Path(class_directory)
#
#         if not class_directory.exists():
#             return 0
#
#         return sum(
#             path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
#             for path in class_directory.iterdir()
#         )
#
#
# class ImageNet21KDataset(Dataset):
#     """One dataset class for primary datasets and subset views."""
#
#     def __init__(
#             self,
#             root=None,
#             classes=None,
#             class_to_name=None,
#             transform=None,
#             base_dataset=None,
#             indices=None,
#     ):
#         self.transform = transform
#
#         if base_dataset is not None:
#             if indices is None:
#                 raise ValueError("indices are required when base_dataset is provided.")
#
#             self.base_dataset = base_dataset
#             self.indices = [int(index) for index in indices]
#
#             self.root = base_dataset.root
#             self.classes = base_dataset.classes
#             self.class_to_idx = base_dataset.class_to_idx
#             self.class_to_name = base_dataset.class_to_name
#             self.dataset_metadata = base_dataset.dataset_metadata
#             self.targets = [base_dataset.targets[index] for index in self.indices]
#             return
#
#         if root is None or classes is None:
#             raise ValueError("root and classes are required.")
#
#         self.base_dataset = None
#         self.indices = None
#         self.root = Path(root)
#         self.classes = list(classes)
#
#         class_to_name = class_to_name or {}
#
#         self.class_to_idx = {
#             wnid: index
#             for index, wnid in enumerate(self.classes)
#         }
#
#         self.class_to_name = {
#             wnid: class_to_name.get(wnid, wnid)
#             for wnid in self.classes
#         }
#
#         self.dataset_metadata = {
#             "dataset": "ImageNet-21K Winter21",
#             "num_classes": len(self.classes),
#         }
#
#         self.samples = []
#
#         for wnid in self.classes:
#             folder = self.root / wnid
#
#             if not folder.exists():
#                 continue
#
#             target = self.class_to_idx[wnid]
#
#             self.samples.extend(
#                 (path, target)
#                 for path in sorted(folder.iterdir())
#                 if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
#             )
#
#         self.targets = [target for _, target in self.samples]
#
#     def subset(self, indices, transform=None):
#         return ImageNet21KDataset(
#             base_dataset=self,
#             indices=indices,
#             transform=transform,
#         )
#
#     def __len__(self):
#         return len(self.indices) if self.base_dataset is not None else len(self.samples)
#
#     def __getitem__(self, index):
#         if self.base_dataset is not None:
#             base_index = self.indices[index]
#             path, target = self.base_dataset.samples[base_index]
#         else:
#             path, target = self.samples[index]
#
#         with Image.open(path) as image:
#             image = image.convert("RGB")
#
#             if self.transform:
#                 image = self.transform(image)
#
#         return image, target
#
#
# class DatasetSplitter:
#     """Creates and caches reproducible stratified dataset splits."""
#
#     def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
#         self.train_ratio, self.val_ratio, self.test_ratio = self._normalize_ratios(
#             train_ratio,
#             val_ratio,
#             test_ratio,
#         )
#         self.seed = seed
#
#     @staticmethod
#     def _normalize_ratios(train_ratio, val_ratio, test_ratio):
#         ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
#
#         if any(ratio < 0 for ratio in ratios):
#             raise ValueError("Split ratios cannot be negative.")
#
#         total = sum(ratios)
#
#         if total <= 0:
#             raise ValueError("At least one split ratio must be positive.")
#
#         return tuple(ratio / total for ratio in ratios)
#
#     def split(self, dataset):
#         rng = random.Random(self.seed)
#         class_indices = defaultdict(list)
#
#         for index, target in enumerate(dataset.targets):
#             class_indices[int(target)].append(index)
#
#         train_indices = []
#         val_indices = []
#         test_indices = []
#
#         for target in sorted(class_indices):
#             indices = class_indices[target].copy()
#             rng.shuffle(indices)
#             n = len(indices)
#
#             if n == 1:
#                 train_indices.extend(indices)
#                 continue
#
#             if n == 2:
#                 train_indices.append(indices[0])
#
#                 if self.val_ratio >= self.test_ratio:
#                     val_indices.append(indices[1])
#                 else:
#                     test_indices.append(indices[1])
#
#                 continue
#
#             n_val = int(round(n * self.val_ratio))
#             n_test = int(round(n * self.test_ratio))
#
#             if self.val_ratio > 0:
#                 n_val = max(1, n_val)
#
#             if self.test_ratio > 0:
#                 n_test = max(1, n_test)
#
#             while n_val + n_test >= n:
#                 if n_val >= n_test and n_val > 0:
#                     n_val -= 1
#                 elif n_test > 0:
#                     n_test -= 1
#                 else:
#                     break
#
#             n_train = n - n_val - n_test
#
#             train_indices.extend(indices[:n_train])
#             val_indices.extend(indices[n_train:n_train + n_val])
#             test_indices.extend(indices[n_train + n_val:])
#
#         rng.shuffle(train_indices)
#         rng.shuffle(val_indices)
#         rng.shuffle(test_indices)
#
#         return train_indices, val_indices, test_indices
#
#     def get_or_create_split(self, dataset, cache_root):
#         cache_root = Path(cache_root)
#         cache_root.mkdir(parents=True, exist_ok=True)
#
#         signature = self._dataset_signature(dataset)
#
#         split_file = cache_root / (
#             f"split_{self.train_ratio:.6f}_{self.val_ratio:.6f}_"
#             f"{self.test_ratio:.6f}_seed{self.seed}_{signature}.json"
#         )
#
#         if split_file.exists():
#             try:
#                 split = json.loads(split_file.read_text(encoding="utf-8"))
#
#                 if (
#                         split.get("dataset_size") == len(dataset)
#                         and split.get("num_classes") == len(dataset.classes)
#                         and split.get("signature") == signature
#                         and split.get("train_ratio") == self.train_ratio
#                         and split.get("val_ratio") == self.val_ratio
#                         and split.get("test_ratio") == self.test_ratio
#                         and split.get("seed") == self.seed
#                 ):
#                     return split["train"], split["val"], split["test"]
#
#             except (json.JSONDecodeError, KeyError, TypeError, ValueError):
#                 pass
#
#             split_file.unlink()
#
#         train_indices, val_indices, test_indices = self.split(dataset)
#
#         split = {
#             "dataset_size": len(dataset),
#             "num_classes": len(dataset.classes),
#             "train_ratio": self.train_ratio,
#             "val_ratio": self.val_ratio,
#             "test_ratio": self.test_ratio,
#             "seed": self.seed,
#             "signature": signature,
#             "train": [int(index) for index in train_indices],
#             "val": [int(index) for index in val_indices],
#             "test": [int(index) for index in test_indices],
#         }
#
#         split_file.write_text(
#             json.dumps(split, indent=2),
#             encoding="utf-8",
#         )
#
#         return split["train"], split["val"], split["test"]
#
#     @staticmethod
#     def _dataset_signature(dataset):
#         value = "|".join(dataset.classes) + f"|{len(dataset)}"
#         return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
#
#
# class ImageNet21KLoader(ImageNetLoader):
#     """Coordinates metadata, downloading, dataset creation and splitting."""
#
#     def __init__(
#             self,
#             mode="sample",
#             root=FULL_ROOT,
#             sample_classes=SAMPLE_NUM_CLASSES,
#             images_per_class=SAMPLE_IMAGES_PER_CLASS,
#             train_ratio=0.8,
#             val_ratio=0.1,
#             test_ratio=0.1,
#             seed=42,
#     ):
#         if mode not in {"sample", "full"}:
#             raise ValueError("mode must be 'sample' or 'full'")
#
#         self.mode = mode
#         self.root = Path(root)
#         self.sample_classes = sample_classes
#         self.images_per_class = images_per_class
#         self.seed = seed
#
#         self.metadata_manager = ImageNetMetadataManager(self.root)
#
#         self.splitter = DatasetSplitter(
#             train_ratio=train_ratio,
#             val_ratio=val_ratio,
#             test_ratio=test_ratio,
#             seed=seed,
#         )
#
#     def load(self):
#         downloader = ImageNet21KDownloader(
#             metadata_manager=self.metadata_manager,
#             mode=self.mode,
#             root=self.root,
#             sample_classes=self.sample_classes,
#             images_per_class=self.images_per_class,
#             seed=self.seed,
#         )
#
#         classes = downloader.download()
#         class_to_name = self.metadata_manager.get_synset_names(classes)
#
#         dataset = ImageNet21KDataset(
#             root=self.root / "images",
#             classes=classes,
#             class_to_name=class_to_name,
#         )
#
#         train_indices, val_indices, test_indices = self.splitter.get_or_create_split(dataset, self.root / "splits")
#         train_dataset = dataset.subset(train_indices, transform=train_transform)
#         val_dataset = dataset.subset(val_indices, transform=test_transform)
#         test_dataset = dataset.subset(test_indices, transform=test_transform)
#         return (train_dataset, val_dataset, test_dataset, "imagenet21k")
#
#
# # imagenet_21k = ImageNet21KLoader(
# #     mode="sample",
# #     sample_classes=200,
# #     images_per_class=None,
# #     train_ratio=0.833,
# #     val_ratio=0.0833,
# #     test_ratio=0.0833,
# #     seed=42,
# # ).load()
#
# # imagenet_21k = model_ready_dataset.ModelReadyDatasetLoader(
# #     ImageNet21KLoader(
# #         mode="sample",
# #         sample_classes=200,
# #         images_per_class=None,
# #         train_ratio=0.833,
# #         val_ratio=0.0833,
# #         test_ratio=0.0833,
# #         seed=42,
# #     )
# # ).load()