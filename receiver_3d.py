import os
import time
from tiled.client.stream import Subscription, LiveArrayData, LiveChildCreated
from tiled.client import from_uri
import time
import pandas
import pyarrow
from concurrent.futures import ThreadPoolExecutor
from automap_hxn.analysis import analyze_data_from_arrays
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from scipy import ndimage as ndi

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --------------------------------------------------------------------------- #
# Config constants (match visualize_3d_tiff defaults)
AXES = {"xy": 0, "xz": 1, "yz": 2}
DEFAULT_CKPT = Path(__file__).parent.parent / "runs/ibm_pcm_ft_rich_prompts/checkpoints/checkpoint.pt"
DEFAULT_VOTE_THRESHOLD = 2

CONF_THRESH = 0.3
DEFAULT_TEXT_PROMPT = "IC feature"
DEFAULT_MIN_COMPONENT_VOXELS = 50
N_SLICE_GRID = 12

# Percentile normalization
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


def run_inference_all_axes(
    ckpt: Path, vol_u8: np.ndarray, text_prompt: str,
) -> dict:
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


# visualization helpers removed — no matplotlib usage required in receiver


URI_IN = os.getenv("URI_IN", "https://tiled.nsls2.bnl.gov/api/v1/metadata/tst/sandbox/eugene/synaps/reconstructions")
URI_OUT = os.getenv("URI_OUT", "https://tiled.nsls2.bnl.gov/api/v1/metadata/tst/sandbox/eugene/synaps/segmentations")

# Cache metadata updates to match them with subsequent data updates.
METADATA_UPDATES = {}
SUBSCRIPTIONS = []

writer_client = from_uri(URI_OUT)
executor = ThreadPoolExecutor(max_workers=4)

def segmentation_function(data, metadata, path_parts):
    """Run SAM3-based 3D segmentation on the incoming volume and upload results.

    The function tries to run the same pipeline used by `visualize_3d_tiff.py`:
    per-axis inference → vote → 3D connected-component labeling. If the
    SAM checkpoint is not available, it falls back to the original
    `analyze_data_from_arrays` behavior.
    """

    print("⏳ Running 3D SAM-based segmentation on new data...")

    dataset_name, _ = path_parts[-2:]
    try:
        container = writer_client[dataset_name]
    except KeyError:
        container = writer_client.create_container(dataset_name, access_tags=["tst_sandbox"])

    # Convert incoming data to a numpy array
    try:
        vol = np.asarray(data)
    except Exception as e:
        print(f"❌ Couldn't convert incoming data to ndarray: {e}")
        return

    # Configurable parameters (env overrides)
    ckpt_path = Path(os.getenv("SAM_CKPT", str(DEFAULT_CKPT)))
    text_prompt = os.getenv("SAM_TEXT_PROMPT", DEFAULT_TEXT_PROMPT)
    vote_threshold = int(os.getenv("SAM_VOTE_THRESHOLD", DEFAULT_VOTE_THRESHOLD))
    min_component_voxels = int(os.getenv("SAM_MIN_COMPONENT_VOXELS", DEFAULT_MIN_COMPONENT_VOXELS))

    # Require a 3D volume for the SAM3 pipeline
    if vol.ndim != 3 or vol.size == 0:
        print(f"⚠️ Expected a 3D volume for SAM3 pipeline, got shape {vol.shape}. Falling back to table-based analyzer.")
        output = analyze_data_from_arrays(vol, metadata)
        if output:
            for channel, boxes in output.items():
                try:
                    if not (table := pyarrow.Table.from_pandas(boxes)):
                        continue
                    table_client = container.create_appendable_table(
                        schema=table.schema,
                        key=channel,
                        metadata=metadata,
                        access_tags=["tst_sandbox"],
                    )
                    time.sleep(0.5)
                    table_client.append_partition(0, table)
                except Exception:
                    import traceback
                    traceback.print_exc()
        else:
            container.write_table({}, key="empty", access_tags=["tst_sandbox"])
        return

    # Run the SAM3 pipeline if checkpoint is available
    if not ckpt_path.exists():
        print(f"⚠️ SAM checkpoint {ckpt_path} not found. Falling back to analyzer.")
        output = analyze_data_from_arrays(vol, metadata)
        if output:
            for channel, boxes in output.items():
                try:
                    if not (table := pyarrow.Table.from_pandas(boxes)):
                        continue
                    table_client = container.create_appendable_table(
                        schema=table.schema,
                        key=channel,
                        metadata=metadata,
                        access_tags=["tst_sandbox"],
                    )
                    time.sleep(0.5)
                    table_client.append_partition(0, table)
                except Exception:
                    import traceback
                    traceback.print_exc()
        else:
            container.write_table({}, key="empty", access_tags=["tst_sandbox"])
        return

    try:
        vol_u8 = norm_to_uint8(vol.astype(np.float32))
        per_axis = run_inference_all_axes(ckpt_path, vol_u8, text_prompt)
        binary = vote(per_axis, vote_threshold)
        labels = stitch_instances(binary, min_component_voxels)

        # Upload per-axis binary predictions
        for name, arr in per_axis.items():
            try:
                container.write_array(arr.astype(np.uint8), key=f"pred_per_axis_{name}", access_tags=["tst_sandbox"])
            except Exception as e:
                print(f"❌ Failed to write per-axis array {name}: {e}")

        # Upload labeled volume
        try:
            container.write_array(labels.astype(np.int32), key="pred_labels3d", access_tags=["tst_sandbox"])
            print("✅ Uploaded predicted labels to Tiled as 'pred_labels3d'.")
        except Exception as e:
            print(f"❌ Failed to upload labeled volume: {e}")

        # No local visualizations rendered in this receiver.

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
    METADATA_UPDATES[path_parts] = update.metadata  # Cache the metadata for later use
    sub = update.child().subscribe()
    sub.child_created.add_callback(on_new_array)
    sub.start_in_thread(start=0)
    SUBSCRIPTIONS.append(sub)


def on_new_array(update: LiveChildCreated):
    "This runs when a new array is created in the container; may not have any data yet!"
    print(f"   New array created: {update.key}. Waiting for data to be uploaded...")
    sub = update.child().subscribe()  # subscribe to the array to get data updates
    sub.new_data.add_callback(run_segmentation)
    sub.start_in_thread(start=0, max_size=100_000_000_000)  # large max_size for bigger images
    SUBSCRIPTIONS.append(sub)


def run_segmentation(update: LiveArrayData):
    "This runs when data is uploaded to the array. The metadata is retrieved from "
    "the cache and passed to the segmentation function."
    path_parts = tuple(update.subscription.segments)
    metadata = METADATA_UPDATES.get(path_parts[:-1], {})  # Get metadata for the parent dataset
    executor.submit(segmentation_function, data=update.data(), metadata=metadata, path_parts=path_parts[-2:])


# To run the function:
if __name__ == "__main__":
    client = from_uri(URI_IN)
    sub = client.subscribe()
    sub.child_created.add_callback(on_new_dataset)
    print("📡 Listening for updates. Use Ctrl+C to stop....", flush=True)
    sub.start()  # block

