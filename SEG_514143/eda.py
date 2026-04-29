# STUDENT's UČO: 514143

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import os

from label_dict import label_dict
from utils import create_dataframe, sample_id_from_name


PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data_seg_public"
IMG_DIR = DATA_DIR / "img"
MASK_DIR = DATA_DIR / "mask"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "eda_outputs"


def bytes_to_mb(size_in_bytes: int) -> float:
	return size_in_bytes / (1024 ** 2)


def bytes_to_gb(size_in_bytes: int) -> float:
	return size_in_bytes / (1024 ** 3)


def tree_size_bytes(root: Path) -> int:
	if not root.exists():
		raise FileNotFoundError(f"Directory does not exist: {root}")
	return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


# --- Faster multiprocessing helpers -------------------------------------------------
WORKER_COLOR_INT_TO_CLASS: dict[int, int] | None = None


def rgb_triplet_to_int(rgb: tuple[int, int, int]) -> int:
	return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def build_color_int_to_class() -> dict[int, int]:
	return {rgb_triplet_to_int(rgb): idx for idx, rgb in enumerate(label_dict.values())}


def init_worker(color_int_to_class: dict[int, int]) -> None:
	"""Initializer for worker processes: sets a module-global mapping.
	This avoids pickling the mapping for every single task.
	"""
	global WORKER_COLOR_INT_TO_CLASS
	WORKER_COLOR_INT_TO_CLASS = color_int_to_class


def count_mask_pixels_by_class_fast(
	mask_path: Path,
	color_int_to_class: dict[int, int],
	num_classes: int,
) -> np.ndarray:
	"""Count pixels per class by encoding RGB into 24-bit integers and using
	NumPy's fast unique/count operations.
	"""
	with Image.open(mask_path) as mask:
		mask_rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)

	# pack RGB to a single 24-bit integer per pixel
	r = mask_rgb[..., 0].astype(np.uint32)
	g = mask_rgb[..., 1].astype(np.uint32)
	b = mask_rgb[..., 2].astype(np.uint32)
	color_ints = (r << 16) | (g << 8) | b

	unique_ints, counts = np.unique(color_ints.reshape(-1), return_counts=True)
	class_counts = np.zeros(num_classes, dtype=np.int64)

	for color_int, cnt in zip(unique_ints, counts, strict=True):
		class_index = color_int_to_class.get(int(color_int))
		if class_index is None:
			raise ValueError(f"Unknown RGB int value {int(color_int)} found in mask: {mask_path}")
		class_counts[class_index] += int(cnt)

	return class_counts


def process_pair(image_path_str: str, mask_path_str: str) -> dict:
	"""Top-level worker function (picklable) that processes one image/mask pair.
	Accepts string paths to minimize pickling overhead.
	"""
	image_path = Path(image_path_str)
	mask_path = Path(mask_path_str)

	image_id = sample_id_from_name(image_path, "_leftImg8bit.png")
	mask_id = sample_id_from_name(mask_path, "_gtFine_color.png")
	if image_id != mask_id:
		raise ValueError(f"Mismatched pair: {image_path.name} vs {mask_path.name}")

	image_size_mb = bytes_to_mb(image_path.stat().st_size)
	mask_size_mb = bytes_to_mb(mask_path.stat().st_size)

	with Image.open(image_path) as image:
		width, height = image.size

	# prefer the worker-global mapping (set by initializer) for speed
	if WORKER_COLOR_INT_TO_CLASS is not None:
		color_map = WORKER_COLOR_INT_TO_CLASS
	else:
		# fallback: build a small mapping locally (single-process mode)
		color_map = build_color_int_to_class()

	class_counts = count_mask_pixels_by_class_fast(
		mask_path=mask_path, color_int_to_class=color_map, num_classes=len(label_dict)
	)

	return {
		"image_size_mb": image_size_mb,
		"mask_size_mb": mask_size_mb,
		"width": int(width),
		"height": int(height),
		"pixels": int(width * height),
		"class_counts": class_counts,
	}


def plot_numeric_histogram(
	values: list[float] | list[int],
	bins: int,
	title: str,
	xlabel: str,
	ylabel: str,
	output_path: Path,
) -> Path:
	plt.figure(figsize=(10, 6))
	plt.hist(values, bins=bins, edgecolor="black", color="#4C78A8")
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path


def plot_class_distribution(
	class_names: list[str],
	class_pixel_counts: np.ndarray,
	output_path: Path,
) -> Path:
	positions = np.arange(len(class_names))
	plt.figure(figsize=(11, 6))
	plt.bar(positions, class_pixel_counts.astype(np.int64), edgecolor="black", color="#59A14F")
	plt.xticks(positions, class_names, rotation=30, ha="right")
	plt.title("Class Distribution From Mask Pixels")
	plt.xlabel("Class")
	plt.ylabel("Pixel count")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()
	return output_path


def main() -> None:
	if not DATA_DIR.exists():
		raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	df = create_dataframe(DATA_DIR)
	image_paths = sorted([Path(path) for path in df["img_path"].tolist()])
	mask_paths = sorted([Path(path) for path in df["mask_path"].tolist()])

	if len(image_paths) != len(mask_paths):
		raise ValueError("The number of images and masks is not equal.")

	class_names = list(label_dict.keys())
	class_pixel_counts = np.zeros(len(class_names), dtype=np.int64)

	widths: list[int] = []
	heights: list[int] = []
	pixels_per_image: list[int] = []
	image_sizes_mb: list[float] = []
	mask_sizes_mb: list[float] = []

	# Build an integer-encoded color->class mapping and use a process pool.
	color_int_to_class = build_color_int_to_class()

	# Determine number of worker processes; keep a safe default but allow override.
	default_workers = max(1, (os.cpu_count() or 1) - 1)
	try:
		env_workers = int(os.environ.get("EDA_WORKERS", "")) if os.environ.get("EDA_WORKERS") else None
	except ValueError:
		env_workers = None
	max_workers = env_workers if env_workers and env_workers > 0 else default_workers

	with concurrent.futures.ProcessPoolExecutor(
		max_workers=max_workers, initializer=init_worker, initargs=(color_int_to_class,)
	) as exe:
		futures = [
			exe.submit(process_pair, str(img_p), str(m_p))
			for img_p, m_p in zip(image_paths, mask_paths)
		]

		for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Computing EDA metrics"):
			res = fut.result()
			image_sizes_mb.append(res["image_size_mb"])
			mask_sizes_mb.append(res["mask_size_mb"])
			widths.append(res["width"])
			heights.append(res["height"])
			pixels_per_image.append(res["pixels"])
			class_pixel_counts += res["class_counts"]

	total_size_gb = bytes_to_gb(tree_size_bytes(IMG_DIR) + tree_size_bytes(MASK_DIR))
	is_uniform = len(set(widths)) == 1 and len(set(heights)) == 1

	print(f"Number of samples: {len(image_paths)}")
	print(f"Total size of img + mask trees: {total_size_gb:.3f} GB")
	print(f"Image size uniform: {is_uniform}")

	generated_files: list[Path] = []
	plot_specs: list[tuple] = [
		(
			plot_class_distribution,
			{
				"class_names": class_names,
				"class_pixel_counts": class_pixel_counts,
				"output_path": OUTPUT_DIR / "class_pixel_distribution_hist.png",
			},
		),
		(
			plot_numeric_histogram,
			{
				"values": image_sizes_mb,
				"bins": 40,
				"title": "Input Image File Size Distribution",
				"xlabel": "Image size (MB)",
				"ylabel": "Count",
				"output_path": OUTPUT_DIR / "image_file_size_mb_hist.png",
			},
		),
		(
			plot_numeric_histogram,
			{
				"values": mask_sizes_mb,
				"bins": 40,
				"title": "Mask File Size Distribution",
				"xlabel": "Mask size (MB)",
				"ylabel": "Count",
				"output_path": OUTPUT_DIR / "mask_file_size_mb_hist.png",
			},
		),
	]

	if is_uniform:
		width_value = widths[0]
		height_value = heights[0]
		print(f"Uniform image width: {width_value}")
		print(f"Uniform image height: {height_value}")
		print(f"Uniform pixels per image: {width_value * height_value}")
		print("Skipped width/height/pixels histograms because image sizes are uniform.")
	else:
		plot_specs.extend(
			[
				(
					plot_numeric_histogram,
					{
						"values": heights,
						"bins": 40,
						"title": "Image Height Distribution",
						"xlabel": "Height (pixels)",
						"ylabel": "Count",
						"output_path": OUTPUT_DIR / "image_heights_hist.png",
					},
				),
				(
					plot_numeric_histogram,
					{
						"values": widths,
						"bins": 40,
						"title": "Image Width Distribution",
						"xlabel": "Width (pixels)",
						"ylabel": "Count",
						"output_path": OUTPUT_DIR / "image_widths_hist.png",
					},
				),
				(
					plot_numeric_histogram,
					{
						"values": pixels_per_image,
						"bins": 40,
						"title": "Pixels Per Image Distribution",
						"xlabel": "Pixels per image",
						"ylabel": "Count",
						"output_path": OUTPUT_DIR / "pixels_per_image_hist.png",
					},
				),
			]
		)

	for plot_func, plot_kwargs in plot_specs:
		generated_files.append(plot_func(**plot_kwargs))

	print("Class pixel counts:")
	for class_name, pixel_count in zip(class_names, class_pixel_counts.tolist(), strict=True):
		print(f"  {class_name}: {pixel_count}")

	print("Saved plots:")
	for output_file in generated_files:
		print(f"  {output_file}")


if __name__ == "__main__":
	main()


