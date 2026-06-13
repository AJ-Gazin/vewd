import json
import re
import shutil
import base64
import io
import struct
import random
import string
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, PngImagePlugin
import folder_paths
from aiohttp import web
from server import PromptServer


# --- Local patch: folder-name token resolution (matches ComfyUI Save Image) ---
# Supports %date:FORMAT% (yyyy/yy, MM/M, dd/d, hh/h, mm/m, ss/s) and the
# simpler %year%/%month%/%day%/%hour%/%minute%/%second% tokens. Not in upstream.
_DATE_RE = re.compile(r"%date:([^%]+)%")
_TOKEN_RE = re.compile(r"(yyyy|yy|MM|M|dd|d|hh|h|mm|m|ss|s)")


def _format_date(fmt: str, now: datetime) -> str:
    def sub(m):
        t = m.group(1)
        if t == "yyyy": return f"{now.year:04d}"
        if t == "yy":   return f"{now.year % 100:02d}"
        if t == "MM":   return f"{now.month:02d}"
        if t == "M":    return f"{now.month}"
        if t == "dd":   return f"{now.day:02d}"
        if t == "d":    return f"{now.day}"
        if t == "hh":   return f"{now.hour:02d}"
        if t == "h":    return f"{now.hour}"
        if t == "mm":   return f"{now.minute:02d}"
        if t == "m":    return f"{now.minute}"
        if t == "ss":   return f"{now.second:02d}"
        if t == "s":    return f"{now.second}"
        return m.group(0)
    return _TOKEN_RE.sub(sub, fmt)


def resolve_folder_tokens(folder: str) -> str:
    if not folder or "%" not in folder:
        return folder
    now = datetime.now()
    folder = _DATE_RE.sub(lambda m: _format_date(m.group(1), now), folder)
    folder = (folder.replace("%year%", f"{now.year:04d}")
                    .replace("%month%", f"{now.month:02d}")
                    .replace("%day%", f"{now.day:02d}")
                    .replace("%hour%", f"{now.hour:02d}")
                    .replace("%minute%", f"{now.minute:02d}")
                    .replace("%second%", f"{now.second:02d}"))
    return folder
# --- end local patch ---

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[Vewd] cv2 not available — video frame extraction disabled, will use screenshot fallback")

# Store latest screenshot per node for IMAGE output (splat fallback)
_screenshot_store = {}

# Store active video file info per node for full-frame extraction
_video_store = {}

# Store active image file info per node for direct-from-disk loading
_image_store = {}

# Store batch of selected media items per node for multi-frame output
_batch_store = {}


def extract_video_frames(video_path, max_frames=0):
    """Read all frames from a video file and return as (N, H, W, 3) float32 tensor."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    # Stack to (N, H, W, 3) float32 normalized
    stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


BINARY_EXTS = {'.glb', '.gltf', '.obj', '.ply', '.splat', '.stl'}
MEDIA_EXTS = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.mp3', '.wav', '.ogg', '.flac', '.aac'}
AUDIO_CONVERT_TO_MP3 = {'.flac', '.wav', '.ogg', '.aac'}  # Convert these to MP3 on save


def copy_with_metadata(src_path, dst_path, seed=None):
    """Copy file, embedding seed as PNG metadata if applicable.
    For audio files in AUDIO_CONVERT_TO_MP3, converts to MP3 via ffmpeg.
    For 3D model files and other non-PNG files, does a plain file copy."""
    ext = Path(src_path).suffix.lower()
    if ext in BINARY_EXTS:
        shutil.copy2(src_path, dst_path)
        return
    # Convert audio to MP3
    if ext in AUDIO_CONVERT_TO_MP3:
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_path), "-codec:a", "libmp3lame", "-q:a", "2", str(dst_path)],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                return
            print(f"[Vewd] ffmpeg conversion failed: {result.stderr.decode()[:200]}")
        except Exception as e:
            print(f"[Vewd] ffmpeg not available, copying as-is: {e}")
        # Fallback: copy original if ffmpeg fails
        shutil.copy2(src_path, Path(str(dst_path).rsplit('.', 1)[0] + ext))
        return
    if ext in MEDIA_EXTS:
        shutil.copy2(src_path, dst_path)
        return
    if seed and ext == '.png':
        try:
            img = Image.open(src_path)
            meta = PngImagePlugin.PngInfo()
            # Preserve existing metadata
            if hasattr(img, 'text'):
                for k, v in img.text.items():
                    meta.add_text(k, v)
            meta.add_text("seed", str(seed))
            img.save(dst_path, pnginfo=meta)
            return
        except Exception:
            pass
    shutil.copy2(src_path, dst_path)

class Vewd:
    """
    Image viewer node with capture mode.
    Captures all images from the workflow automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input": ("IMAGE",),
                # --- Local patch: extra accumulate sockets (not in upstream) ---
                # In capture="input" mode every connected socket becomes its own grid
                # tile (any resolution). e.g. wire original -> input, upscaled -> input_2.
                "input_2": ("IMAGE",),
                "input_3": ("IMAGE",),
                "input_4": ("IMAGE",),
                # --- end local patch ---
                "folder": ("STRING", {"default": "C:/AI/comfy/ComfyUI/output/vewd"}),
                "filename_prefix": ("STRING", {"default": "vewd"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "Max video frames to extract (0 = all)"}),
                # --- Local patch: capture mode (not in upstream) ---
                # auto  = grab every image-emitting node in the workflow (whole-workflow scrape; matches vewdtab)
                # input = show only the tensor wired into this node's `input` socket
                "capture": (["auto", "input"], {"default": "auto", "tooltip": "auto = capture all workflow images; input = only the wired input"}),
                # --- end local patch ---
                "selected_media": ("STRING", {"default": ""}),
                # --- Local patch: wire-driven prefix (not in upstream) ---
                # MUST stay last: appending keeps positional widget values of older
                # saved workflows aligned (inserting mid-list shifts them and breaks
                # max_frames/capture). Optional STRING socket — when connected, its
                # value (e.g. a shortened model name) is pushed into the grid's prefix
                # field at run time so saved files use it.
                "prefix": ("STRING", {"forceInput": True}),
                # --- end local patch ---
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "image"
    # --- Local patch: mark as a terminal node (not in upstream) ---
    # ComfyUI prunes any branch that doesn't reach an OUTPUT_NODE. Without this,
    # wiring a node solely into Vewd's input leaves that branch with no consumer,
    # so its upstream (e.g. Ultimate SD Upscale) never executes. Marking Vewd as an
    # output node forces its wired inputs to run, same as Preview Image / Save Image.
    OUTPUT_NODE = True
    # --- end local patch ---

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def process(self, input=None, input_2=None, input_3=None, input_4=None, folder="", filename_prefix="vewd", prefix=None, max_frames=0, capture="auto", selected_media="", prompt=None, extra_pnginfo=None, unique_id=None):
        folder = folder.strip('"')
        result = {"ui": {"vewd_images": []}}
        img_tensor = None
        node_key = str(unique_id) if unique_id else None

        # Local patch: all connected accumulate sockets, in order (used by capture="input")
        wired_inputs = [t for t in (input, input_2, input_3, input_4) if t is not None]

        # Priority: wired input > selected_media widget > video store > image store > screenshot store > black fallback
        # Passthrough output uses the first connected socket (keeps downstream stable).
        if wired_inputs:
            img_tensor = wired_inputs[0]

        # Parse selected_media widget (cloud-compatible passthrough)
        # Supports array of items (multi-select → batch tensor) or single object (legacy)
        if img_tensor is None and selected_media:
            try:
                parsed = json.loads(selected_media)
                # Normalize to list
                media_list = parsed if isinstance(parsed, list) else [parsed]

                type_dirs = {
                    "temp": folder_paths.get_temp_directory(),
                    "output": folder_paths.get_output_directory(),
                    "input": folder_paths.get_input_directory(),
                }

                loaded_tensors = []
                target_size = None

                for media_info in media_list:
                    media_type = media_info.get("media_type", "")
                    filename = media_info.get("filename", "")
                    subfolder = media_info.get("subfolder", "")
                    source_type = media_info.get("type", "temp")

                    if not filename:
                        continue

                    base_dir = type_dirs.get(source_type, folder_paths.get_temp_directory())
                    file_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename

                    if not file_path.exists():
                        print(f"[Vewd] Widget: file not found: {file_path}")
                        continue

                    if media_type == "video" and HAS_CV2:
                        frames = extract_video_frames(file_path, max_frames)
                        print(f"[Vewd] Widget: extracted {frames.shape[0]} frames from {file_path.name}")
                        loaded_tensors.append(frames)
                    elif media_type == "image" or file_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'):
                        img = Image.open(file_path).convert("RGB")
                        # Resize to match first image (batch tensors must be same size)
                        if target_size is None:
                            target_size = img.size
                        elif img.size != target_size:
                            img = img.resize(target_size, Image.LANCZOS)
                        img_array = np.array(img).astype(np.float32) / 255.0
                        loaded_tensors.append(torch.from_numpy(img_array).unsqueeze(0))

                if loaded_tensors:
                    img_tensor = torch.cat(loaded_tensors, dim=0)
                    print(f"[Vewd] Widget: batch of {img_tensor.shape[0]} frames ({img_tensor.shape[2]}x{img_tensor.shape[1]})")

            except Exception as e:
                print(f"[Vewd] Widget: selected_media parse failed: {e}")

        # Batch store — multiple selected items via HTTP endpoint
        if img_tensor is None and node_key and node_key in _batch_store:
            batch_items = _batch_store[node_key]
            type_dirs = {
                "temp": folder_paths.get_temp_directory(),
                "output": folder_paths.get_output_directory(),
                "input": folder_paths.get_input_directory(),
            }
            loaded_tensors = []
            target_size = None
            for item in batch_items:
                media_type = item.get("media_type", "")
                filename = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                source_type = item.get("type", "temp")
                if not filename:
                    continue
                base_dir = type_dirs.get(source_type, folder_paths.get_temp_directory())
                file_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename
                if not file_path.exists():
                    print(f"[Vewd] Batch: file not found: {file_path}")
                    continue
                if media_type == "video" and HAS_CV2:
                    frames = extract_video_frames(file_path, max_frames)
                    loaded_tensors.append(frames)
                elif media_type == "image" or file_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'):
                    img = Image.open(file_path).convert("RGB")
                    if target_size is None:
                        target_size = img.size
                    elif img.size != target_size:
                        img = img.resize(target_size, Image.LANCZOS)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    loaded_tensors.append(torch.from_numpy(img_array).unsqueeze(0))
            if loaded_tensors:
                img_tensor = torch.cat(loaded_tensors, dim=0)
                print(f"[Vewd] Batch: {img_tensor.shape[0]} frames ({img_tensor.shape[2]}x{img_tensor.shape[1]})")
            # Clear batch after use so single-select works next time
            del _batch_store[node_key]

        if img_tensor is None and node_key and node_key in _video_store and HAS_CV2:
            video_info = _video_store[node_key]
            try:
                # Resolve video file path
                type_dirs = {
                    "temp": folder_paths.get_temp_directory(),
                    "output": folder_paths.get_output_directory(),
                    "input": folder_paths.get_input_directory(),
                }
                base_dir = type_dirs.get(video_info.get("type", "temp"), folder_paths.get_temp_directory())
                subfolder = video_info.get("subfolder", "")
                filename = video_info["filename"]
                video_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename

                if video_path.exists():
                    img_tensor = extract_video_frames(video_path, max_frames)
                    print(f"[Vewd] Extracted {img_tensor.shape[0]} frames from {video_path.name}")
                else:
                    print(f"[Vewd] Video file not found: {video_path}")
            except Exception as e:
                print(f"[Vewd] Video extraction failed: {e}")

        if img_tensor is None and node_key and node_key in _image_store:
            image_info = _image_store[node_key]
            try:
                type_dirs = {
                    "temp": folder_paths.get_temp_directory(),
                    "output": folder_paths.get_output_directory(),
                    "input": folder_paths.get_input_directory(),
                }
                base_dir = type_dirs.get(image_info.get("type", "temp"), folder_paths.get_temp_directory())
                subfolder = image_info.get("subfolder", "")
                filename = image_info["filename"]
                img_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename

                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    print(f"[Vewd] Loaded image from disk: {img_path.name} ({img.size[0]}x{img.size[1]})")
                else:
                    print(f"[Vewd] Image file not found: {img_path}")
            except Exception as e:
                print(f"[Vewd] Image load failed: {e}")

        if img_tensor is None and node_key and node_key in _screenshot_store:
            img_data = _screenshot_store[node_key]
            try:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            except Exception as e:
                print(f"[Vewd] Failed to load screenshot: {e}")

        # Legacy fallback: try any screenshot if node-specific not found
        if img_tensor is None and _screenshot_store:
            latest_key = max(_screenshot_store.keys())
            img_data = _screenshot_store[latest_key]
            try:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            except Exception as e:
                print(f"[Vewd] Failed to load screenshot: {e}")

        # Always return an image tensor (black 512x512 fallback)
        if img_tensor is None:
            img_tensor = torch.zeros(1, 512, 512, 3)

        # --- Local patch: capture="input" — surface the wired tensor(s) as grid previews ---
        # Saves every connected accumulate socket (each frame of each), so original +
        # upscaled land as separate tiles at their own resolutions. Only emits when
        # something is wired in (skips the black fallback). The frontend listener drops
        # every other node's output while in "input" mode, so these are the sole sources.
        if capture == "input" and wired_inputs:
            try:
                temp_dir = folder_paths.get_temp_directory()
                Path(temp_dir).mkdir(parents=True, exist_ok=True)
                token = ''.join(random.choices(string.ascii_lowercase, k=5))

                # Embed prompt + workflow into the PNG, same as ComfyUI's SaveImage.
                # Without this the input-mode temp file has no graph, so a later save
                # produces a seed-only PNG (no prompt/workflow for metadata readers).
                metadata = None
                try:
                    from comfy.cli_args import args as _comfy_args
                    disable_meta = getattr(_comfy_args, "disable_metadata", False)
                except Exception:
                    disable_meta = False
                if not disable_meta:
                    metadata = PngImagePlugin.PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for k, v in extra_pnginfo.items():
                            metadata.add_text(k, json.dumps(v))

                ui_images = []
                idx = 0
                for tensor in wired_inputs:
                    for i in range(tensor.shape[0]):
                        arr = (tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        fname = f"vewd_input_{token}_{idx:03d}.png"
                        Image.fromarray(arr).save(Path(temp_dir) / fname, pnginfo=metadata)
                        ui_images.append({"filename": fname, "subfolder": "", "type": "temp"})
                        idx += 1
                # Emit under a private key (not "images") so ComfyUI doesn't draw its
                # native under-node preview — the Vewd grid is the only consumer.
                result["ui"] = {"vewd_images": ui_images}
            except Exception as e:
                print(f"[Vewd] input-mode preview save failed: {e}")
        # --- end local patch ---

        # --- Local patch: push wired prefix to the grid's prefix field ---
        # Wrap in a list: ComfyUI expects ui values to be lists and otherwise coerces
        # a bare string into list(str) — i.e. one element per character ("o","n",...).
        if prefix is not None and str(prefix).strip():
            result.setdefault("ui", {})["vewd_prefix"] = [str(prefix).strip()]
        # --- end local patch ---

        result["result"] = (img_tensor,)
        return result


# Export API route
@PromptServer.instance.routes.post("/vewd/export")
async def export_selects(request):
    try:
        data = await request.json()
        folder = resolve_folder_tokens(data.get("folder", "").strip('"'))
        prefix = data.get("prefix", "select")
        images = data.get("images", [])

        if not folder or not images:
            return web.json_response({"success": False, "error": "Missing folder or images"})

        # Create selects subfolder
        selects_dir = Path(folder) / "selects"
        selects_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = folder_paths.get_temp_directory()
        output_dir = folder_paths.get_output_directory()
        input_dir = folder_paths.get_input_directory()
        count = 0
        debug = []

        for i, img_info in enumerate(images):
            # Support both old format (string) and new format (object with source info)
            if isinstance(img_info, str):
                filename = img_info
                subfolder = ""
                source_type = "temp"
            else:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                source_type = img_info.get("type", "temp")
                seed = img_info.get("seed")

            # Resolve source path based on type
            type_dirs = {"temp": temp_dir, "output": output_dir, "input": input_dir}
            base_dir = type_dirs.get(source_type, temp_dir)
            src_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename
            tried = [str(src_path)]
            if not src_path.exists():
                src_path = Path(folder) / filename
                tried.append(str(src_path))

            if src_path.exists():
                orig_ext = Path(filename).suffix.lower()
                if orig_ext in BINARY_EXTS:
                    save_ext = orig_ext
                elif orig_ext in AUDIO_CONVERT_TO_MP3:
                    save_ext = ".mp3"
                elif orig_ext in MEDIA_EXTS:
                    save_ext = orig_ext
                else:
                    save_ext = ".png"
                if seed:
                    new_name = f"{prefix}_{seed}_{i + 1:03d}{save_ext}"
                else:
                    orig_stem = Path(filename).stem
                    new_name = f"{prefix}_{orig_stem}{save_ext}"
                dst_path = selects_dir / new_name
                copy_with_metadata(src_path, dst_path, seed)
                count += 1
            else:
                debug.append({"filename": filename, "type": source_type, "subfolder": subfolder, "tried": tried})

        return web.json_response({"success": True, "count": count, "folder": str(selects_dir), "debug": debug})

    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


# Save API route (saves to folder directly, not /selects)
@PromptServer.instance.routes.post("/vewd/save")
async def save_images(request):
    try:
        data = await request.json()
        folder = resolve_folder_tokens(data.get("folder", "").strip('"'))
        prefix = data.get("prefix", "vewd")
        images = data.get("images", [])

        if not folder or not images:
            return web.json_response({"success": False, "error": "Missing folder or images"})

        save_dir = Path(folder)
        save_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = folder_paths.get_temp_directory()
        output_dir = folder_paths.get_output_directory()
        input_dir = folder_paths.get_input_directory()
        count = 0

        # --- Local patch: name saved files prefix_NNN (no seed in filename) ---
        # Continue numbering from the highest existing {prefix}_NNN{ext} in the folder.
        # The seed is still embedded in PNG metadata via copy_with_metadata below.
        prefix_counters = {}

        def next_number(pfx, ext):
            key = (pfx, ext)
            if key not in prefix_counters:
                pat = re.compile(re.escape(pfx) + r"_(\d+)" + re.escape(ext) + r"$", re.IGNORECASE)
                highest = 0
                for f in save_dir.glob(f"{pfx}_*{ext}"):
                    m = pat.match(f.name)
                    if m:
                        highest = max(highest, int(m.group(1)))
                prefix_counters[key] = highest
            prefix_counters[key] += 1
            return prefix_counters[key]
        # --- end local patch ---

        for img_info in images:
            if isinstance(img_info, str):
                filename = img_info
                subfolder = ""
                source_type = "temp"
                seed = None
            else:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                source_type = img_info.get("type", "temp")
                seed = img_info.get("seed")

            type_dirs = {"temp": temp_dir, "output": output_dir, "input": input_dir}
            base_dir = type_dirs.get(source_type, temp_dir)
            src_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename
            if not src_path.exists():
                src_path = Path(folder) / filename

            if src_path.exists():
                orig_ext = Path(filename).suffix.lower()
                if orig_ext in BINARY_EXTS:
                    save_ext = orig_ext
                elif orig_ext in AUDIO_CONVERT_TO_MP3:
                    save_ext = ".mp3"
                elif orig_ext in MEDIA_EXTS:
                    save_ext = orig_ext
                else:
                    save_ext = ".png"
                # Local patch: prefix_NNN naming (seed kept in PNG metadata, not filename)
                new_name = f"{prefix}_{next_number(prefix, save_ext):03d}{save_ext}"
                dst_path = save_dir / new_name
                copy_with_metadata(src_path, dst_path, seed)
                count += 1

        return web.json_response({"success": True, "count": count, "folder": str(save_dir)})

    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


# Screenshot upload endpoint — stores base64 PNG for IMAGE output
@PromptServer.instance.routes.post("/vewd/screenshot")
async def upload_screenshot(request):
    try:
        data = await request.json()
        image_data = data.get("image", "")
        node_id = data.get("node_id", "default")

        # Strip data URL prefix
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        _screenshot_store[node_id] = img_bytes

        # Non-video content is now active — clear video store for this node
        _video_store.pop(node_id, None)

        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


# Batch selection endpoint — stores multiple selected media items for batch output
@PromptServer.instance.routes.post("/vewd/set_batch")
async def set_batch(request):
    try:
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        items = data.get("items", [])

        if not node_id:
            return web.json_response({"success": False, "error": "Missing node_id"})

        if not items:
            _batch_store.pop(node_id, None)
            return web.json_response({"success": True, "cleared": True})

        _batch_store[node_id] = items
        return web.json_response({"success": True, "count": len(items)})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


# Video file info endpoint — stores which video file is active for frame extraction
@PromptServer.instance.routes.post("/vewd/set_video")
async def set_video(request):
    try:
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        filename = data.get("filename", "")

        if not node_id:
            return web.json_response({"success": False, "error": "Missing node_id"})

        if not filename:
            # Clear video info for this node
            _video_store.pop(node_id, None)
            return web.json_response({"success": True, "cleared": True})

        _video_store[node_id] = {
            "filename": filename,
            "subfolder": data.get("subfolder", ""),
            "type": data.get("type", "temp"),
        }

        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


# Image file info endpoint — stores which image is selected for direct loading
@PromptServer.instance.routes.post("/vewd/set_image")
async def set_image(request):
    try:
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        filename = data.get("filename", "")

        if not node_id:
            return web.json_response({"success": False, "error": "Missing node_id"})

        if not filename:
            _image_store.pop(node_id, None)
            return web.json_response({"success": True, "cleared": True})

        _image_store[node_id] = {
            "filename": filename,
            "subfolder": data.get("subfolder", ""),
            "type": data.get("type", "temp"),
        }

        # Image selected — clear video store for this node
        _video_store.pop(node_id, None)

        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


WAVEFORM_COLORS = [
    (70, 130, 200),   # blue
    (200, 75, 100),   # pink
    (70, 190, 120),   # green
    (200, 160, 60),   # gold
    (140, 90, 200),   # purple
    (200, 110, 50),   # orange
    (60, 170, 170),   # cyan
    (200, 65, 65),    # red
]


def generate_waveform(audio_path, width=256, height=256):
    """Generate a waveform thumbnail image from an audio file.
    Uses ffmpeg to decode to raw PCM, then draws the waveform.
    Color is deterministic per filename for consistency."""
    try:
        import subprocess
        # Decode audio to raw 16-bit mono PCM via ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-f", "s16le", "-ac", "1", "-ar", "22050", "-"],
            capture_output=True, timeout=15
        )
        if result.returncode != 0:
            return None
        raw = result.stdout
        if len(raw) < 4:
            return None
        # Parse PCM samples
        n_samples = len(raw) // 2
        samples = np.array(struct.unpack(f"<{n_samples}h", raw[:n_samples * 2]), dtype=np.float32)
        # Normalize
        peak = np.max(np.abs(samples)) or 1.0
        samples = samples / peak
        # Downsample to width bins
        bin_size = max(1, len(samples) // width)
        bins = len(samples) // bin_size
        trimmed = samples[:bins * bin_size].reshape(bins, bin_size)
        maxes = np.max(trimmed, axis=1)
        mins = np.min(trimmed, axis=1)
        # Pick color based on filename hash
        color = WAVEFORM_COLORS[hash(audio_path.name) % len(WAVEFORM_COLORS)]
        # Draw
        img = Image.new("RGB", (width, height), (17, 17, 17))
        draw = ImageDraw.Draw(img)
        mid = height // 2
        for x in range(min(bins, width)):
            y_top = int(mid - maxes[x] * mid * 0.75)
            y_bot = int(mid - mins[x] * mid * 0.75)
            draw.line([(x, y_top), (x, y_bot)], fill=color)
        # Center line
        draw.line([(0, mid), (width - 1, mid)], fill=(40, 40, 40))
        return img
    except Exception as e:
        print(f"[Vewd] Waveform generation failed: {e}")
        return None


@PromptServer.instance.routes.get("/vewd/waveform")
async def get_waveform(request):
    """Generate and return a waveform thumbnail for an audio file.
    Caches the result in output/vewd-cache/ for persistence across restarts."""
    try:
        filename = request.query.get("filename", "")
        subfolder = request.query.get("subfolder", "")
        source_type = request.query.get("type", "temp")

        if not filename:
            return web.json_response({"error": "Missing filename"}, status=400)

        # Check cache first
        cache_dir = Path(folder_paths.get_output_directory()) / "vewd-cache"
        cache_name = Path(filename).stem + "_waveform.png"
        cache_path = cache_dir / cache_name

        if cache_path.exists():
            return web.Response(body=cache_path.read_bytes(), content_type="image/png")

        # Find source audio
        type_dirs = {
            "temp": folder_paths.get_temp_directory(),
            "output": folder_paths.get_output_directory(),
            "input": folder_paths.get_input_directory(),
        }
        base_dir = type_dirs.get(source_type, folder_paths.get_temp_directory())
        audio_path = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename

        if not audio_path.exists():
            # Try alternate location
            alt_type = "output" if source_type == "temp" else "temp"
            alt_dir = type_dirs.get(alt_type, folder_paths.get_temp_directory())
            audio_path = Path(alt_dir) / subfolder / filename if subfolder else Path(alt_dir) / filename

        if not audio_path.exists():
            return web.json_response({"error": "File not found"}, status=404)

        img = generate_waveform(audio_path, width=800, height=200)
        if img is None:
            return web.json_response({"error": "Waveform generation failed"}, status=500)

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        img.save(cache_path, format="PNG")

        # Also cache the source audio file for playback after temp is cleared
        audio_cache_path = cache_dir / Path(filename).name
        if not audio_cache_path.exists():
            try:
                shutil.copy2(audio_path, audio_cache_path)
            except Exception as e:
                print(f"[Vewd] Audio cache failed: {e}")

        return web.Response(body=cache_path.read_bytes(), content_type="image/png")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/vewd/find_latest")
async def find_latest(request):
    """Find the latest file matching a prefix in the output directory.
    Used for nodes like ACE-Step that save files but don't expose them via executed event."""
    try:
        prefix = request.query.get("prefix", "")
        if not prefix:
            return web.json_response({"error": "Missing prefix"}, status=400)

        output_dir = Path(folder_paths.get_output_directory())
        # prefix could be "ace-step/text2music" → search in output/ace-step/ for text2music*
        if "/" in prefix:
            parts = prefix.rsplit("/", 1)
            search_dir = output_dir / parts[0]
            file_prefix = parts[1]
        else:
            search_dir = output_dir
            file_prefix = prefix

        if not search_dir.exists():
            return web.json_response({"error": "Directory not found"}, status=404)

        # Find newest file matching prefix
        audio_exts = {'.mp3', '.wav', '.ogg', '.flac', '.aac'}
        matches = [f for f in search_dir.iterdir()
                   if f.is_file() and f.name.startswith(file_prefix) and f.suffix.lower() in audio_exts]

        if not matches:
            return web.json_response({"error": "No matching files"}, status=404)

        latest = max(matches, key=lambda f: f.stat().st_mtime)
        # Return path relative to output dir
        rel_path = latest.relative_to(output_dir)
        subfolder = str(rel_path.parent) if rel_path.parent != Path(".") else ""

        return web.json_response({
            "filename": latest.name,
            "subfolder": subfolder.replace("\\", "/"),
            "type": "output"
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


NODE_CLASS_MAPPINGS = {
    "Vewd": Vewd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vewd": "Vewd",
}
