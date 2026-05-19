import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from tiled.client.stream import LiveArrayData, LiveChildCreated
from tiled.client import from_uri

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ACCESS_TAG: str = "hxn_processed"
AXES = {"xy": 0, "xz": 1, "yz": 2}
DEFAULT_CKPT = "/nsls2/data2/hxn/legacy/home/home/SYNAPS/hgoel1/sam3/runs/ibm_pcm_ft_rich_prompts/checkpoints/checkpoint.pt"
DEFAULT_VOTE_THRESHOLD = 2
DEFAULT_TEXT_PROMPT = "IC feature"
DEFAULT_MIN_COMPONENT_VOXELS = 15
CONF_THRESH = 0.3
NORM_LO_PCT, NORM_HI_PCT = 1.0, 99.9


def norm_to_uint8(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, (NORM_LO_PCT, NORM_HI_PCT))
    return np.clip(255 * (vol - lo) / max(hi - lo, 1e-12), 0, 255).astype(np.uint8)


def make_rgb_frame(vol_u8: np.ndarray, axis: int, i: int) -> Image.Image:
    n = vol_u8.shape[axis]
    prev_i = max(0, i - 1)
    next_i = min(n - 1, i + 1)
    r = np.take(vol_u8, prev_i, axis=axis)
    g = np.take(vol_u8, i, axis=axis)
    b = np.take(vol_u8, next_i, axis=axis)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _predict_frame(processor: Sam3Processor, img: Image.Image, text_prompt: str) -> np.ndarray:
    state = processor.set_image(img)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt=text_prompt, state=state)
    H, W = img.height, img.width
    out = np.zeros((H, W), dtype=bool)
    if "masks" not in state or state["masks"] is None:
        return out
    pm = state["masks"]
    if torch.is_tensor(pm):
        pm = pm.cpu().numpy()
    for j in range(pm.shape[0]):
        m = pm[j]
        while m.ndim > 2:
            m = m[0] if m.shape[0] == 1 else m.any(axis=0)
        out |= m.astype(bool)
    return out


def run_axis_inference(
    processor: Sam3Processor,
    vol_u8: np.ndarray,
    axis_name: str,
    axis: int,
    text_prompt: str,
) -> np.ndarray:
    n_frames = vol_u8.shape[axis]
    vol = np.zeros(vol_u8.shape, dtype=np.uint8)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n_frames):
            mask = _predict_frame(processor, make_rgb_frame(vol_u8, axis, i), text_prompt)
            sl = [slice(None)] * 3
            sl[axis] = i
            vol[tuple(sl)] = mask.astype(np.uint8)
            if (i + 1) % 50 == 0 or i == n_frames - 1:
                print(f"    [{axis_name}] {i+1}/{n_frames}  ({time.time()-t0:.1f}s)")
    return vol


def run_inference_all_axes(ckpt: Path, vol_u8: np.ndarray, text_prompt: str) -> dict:
    print(f"Loading model from {ckpt} …")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt), load_from_HF=False,
        enable_segmentation=True, device="cuda", eval_mode=True,
    )
    processor = Sam3Processor(model, confidence_threshold=CONF_THRESH)
    print(f"  volume: {vol_u8.shape}")
    print(f"  text prompt: {text_prompt!r}")

    out = {}
    for name, axis in AXES.items():
        print(f"\n  [{name}] axis={axis}")
        out[name] = run_axis_inference(processor, vol_u8, name, axis, text_prompt)
    return out


def vote(per_axis: dict, threshold: int) -> np.ndarray:
    stack = np.stack(list(per_axis.values()), axis=0).astype(np.uint8)
    votes = stack.sum(axis=0)
    for k in range(1, 4):
        print(f"  voxels with ≥{k} axis agreement: "
              f"{int((votes >= k).sum()):>10}  "
              f"({100*(votes >= k).mean():.2f}%)")
    return (votes >= threshold).astype(np.uint8)


def stitch_instances(binary_vol: np.ndarray, min_voxels: int) -> np.ndarray:
    print("3D connected-component labeling …")
    lbl, n = ndi.label(binary_vol, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    remap = np.zeros_like(sizes)
    remap[keep] = np.arange(1, keep.sum() + 1)
    labels = remap[lbl].astype(np.int32)
    print(f"  {n} raw components → {int(labels.max())} after size filter "
          f"(min={min_voxels} vox)  fg%={100*(labels>0).mean():.2f}")
    return labels


def make_thumbnail(labels: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Fast grayscale isometric-ish thumbnail from a 3-D label volume.

    Three binary max-projections (top / front / side) are arranged into a
    single L-shaped composite image.  All work is three np.max calls plus
    array slicing – no interpolation or rotation.

    Layout (nz = depth, ny = height, nx = width)::

        +--------+--------+
        | (empty)| top    |  <- looking down  (ny × nx)
        +--------+--------+
        | side   | front  |  <- side (nz × ny)  front (nz × nx)
        +--------+--------+
    """
    binary = (labels > 0).astype(np.uint8)  # 0/1, shape (nz, ny, nx)
    nz, ny, nx = binary.shape
    top   = binary.max(axis=0)              # (ny, nx)  – looking down
    front = binary.max(axis=1)              # (nz, nx)  – looking from front
    side  = binary.max(axis=2)              # (nz, ny)  – looking from side

    canvas = np.zeros((ny + nz, ny + nx), dtype=np.uint8)
    canvas[:ny, ny:]  = top    # upper-right
    canvas[ny:, :ny]  = side   # lower-left
    canvas[ny:, ny:]  = front  # lower-right

    # Nearest-neighbour downscale to max_side (pure numpy, no PIL)
    h, w = canvas.shape
    scale = max_side / max(h, w)
    if scale < 1.0:
        rr = (np.arange(int(h * scale)) * h // int(h * scale)).astype(np.intp)
        cc = (np.arange(int(w * scale)) * w // int(w * scale)).astype(np.intp)
        canvas = canvas[np.ix_(rr, cc)]

    return (canvas * 255).astype(np.uint8)


URI_IN = os.getenv("URI_IN", "https://tiled.nsls2.bnl.gov/api/v1/metadata/hxn/processed/reconstructions")
URI_OUT = os.getenv("URI_OUT", "https://tiled.nsls2.bnl.gov/api/v1/metadata/hxn/processed/segmentations")

METADATA_UPDATES = {}
SUBSCRIPTIONS = []
PROCESSING = set()

api_key = os.getenv("API_KEY")
reader_client = from_uri(URI_IN, api_key=api_key)
writer_client = from_uri(URI_OUT, api_key=api_key)
executor = ThreadPoolExecutor(max_workers=4)


def _on_future_done(dataset_name, future):
    PROCESSING.discard(dataset_name)
    exc = future.exception()
    if exc is not None:
        import traceback
        print(f"❌ Exception in segmentation for {dataset_name}:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def segmentation_function(data, metadata, path_parts):
    print("⏳ Running 3D SAM-based segmentation on new data...")

    dataset_name, _ = path_parts[-2:]
    try:
        container = writer_client[dataset_name]
    except:
        container = writer_client.create_container(dataset_name, access_tags=[ACCESS_TAG])

    try:
        vol = np.asarray(data)
    except Exception as e:
        print(f"❌ Couldn't convert incoming data to ndarray: {e}")
        return

    ckpt_path = Path(os.getenv("SAM_CKPT", DEFAULT_CKPT))
    text_prompt = metadata.get("text_prompt", DEFAULT_TEXT_PROMPT)
    vote_threshold = int(float(metadata.get("vote_threshold", DEFAULT_VOTE_THRESHOLD)))
    min_component_voxels = int(float(metadata.get("min_component_voxels", DEFAULT_MIN_COMPONENT_VOXELS)))

    if vol.ndim != 3 or vol.size == 0:
        print(f"⚠️ Expected a 3D volume, got shape {vol.shape}. Skipping.")
        return

    if not ckpt_path.exists():
        print(f"⚠️ SAM checkpoint {ckpt_path} not found. Skipping.")
        return

    try:
        vol_u8 = norm_to_uint8(vol.astype(np.float32))
        per_axis = run_inference_all_axes(ckpt_path, vol_u8, text_prompt)
        binary = vote(per_axis, vote_threshold)
        labels = stitch_instances(binary, min_component_voxels)

        for name, arr in per_axis.items():
            try:
                container.write_array(arr.astype(np.uint8), key=f"pred_per_axis_{name}", access_tags=[ACCESS_TAG])
            except Exception as e:
                print(f"❌ Failed to write per-axis array {name}: {e}")

        try:
            container.write_array(labels.astype(np.int32), key="pred_labels3d", access_tags=[ACCESS_TAG])
            print("✅ Uploaded predicted labels to Tiled as 'pred_labels3d'.")
        except Exception as e:
            print(f"❌ Failed to upload labeled volume: {e}")

        try:
            thumb = make_thumbnail(labels)
            container.write_array(thumb, key="thumbnail", access_tags=[ACCESS_TAG])
            print(f"✅ Uploaded thumbnail to Tiled as 'thumbnail' ({thumb.shape[0]}×{thumb.shape[1]} px).")
        except Exception as e:
            print(f"❌ Failed to upload thumbnail: {e}")

        print(f"✅ SAM3 segmentation complete for dataset {dataset_name}: {int(labels.max())} instances.")

    except Exception as e:
        print(f"❌ SAM3 pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def on_new_dataset(update: LiveChildCreated):
    "This runs when a new dataset is created in the root container."
    path_parts = tuple(update.subscription.segments) + (update.key,)
    print(f"\n✨ New dataset created: {'/'.join(path_parts[-3:])}")
    print("   Subscribing to updates...")
    METADATA_UPDATES[path_parts] = update.metadata
    sub = update.child().subscribe()
    sub.child_created.add_callback(on_new_array)
    sub.start_in_thread(start=0)
    SUBSCRIPTIONS.append(sub)


def on_new_array(update: LiveChildCreated):
    "This runs when a new array is created in the container; may not have any data yet!"
    print(f"   New array created: {update.key}. Waiting for data to be uploaded...")
    sub = update.child().subscribe()
    sub.new_data.add_callback(run_segmentation)
    sub.start_in_thread(start=0, max_size=100_000_000_000)
    SUBSCRIPTIONS.append(sub)


def run_segmentation(update: LiveArrayData):
    "Runs when data is uploaded to the array. Read the full array from the server instead of using the streamed chunk."
    path_parts = tuple(update.subscription.segments)
    dataset_name = path_parts[-2]

    if dataset_name in PROCESSING:
        print(f"⏭️ Skipping chunk event for {dataset_name} (already processing)")
        return
    PROCESSING.add(dataset_name)

    metadata = METADATA_UPDATES.get(path_parts[:-1], {})
    array_name = path_parts[-1]
    data = np.asarray(reader_client[dataset_name][array_name].read())
    future = executor.submit(segmentation_function, data=data, metadata=metadata, path_parts=path_parts[-2:])
    future.add_done_callback(lambda f: _on_future_done(dataset_name, f))


if __name__ == "__main__":
    client = from_uri(URI_IN, api_key=api_key)
    sub = client.subscribe()
    sub.child_created.add_callback(on_new_dataset)
    print("📡 Listening for updates. Use Ctrl+C to stop....", flush=True)
    sub.start()
