import math
import os
import numpy as np
from PIL import Image

def _row_entropy(data: np.ndarray, row: int) -> int:

    n = len(data)
    center = row * 256
    start  = max(0, center - 128)
    end    = min(n, center + 128)
    chunk  = data[start:end]
    if len(chunk) == 0:
        return 0
    counts = np.bincount(chunk, minlength=256)
    probs  = counts / len(chunk)
    probs  = probs[probs > 0]
    if len(probs) == 0:
        return 0
    h = -np.sum(probs * np.log2(probs))  
    return int(h / 8.0 * 255)


def _build_entropy_array(filepath: str) -> np.ndarray:

    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(data) == 0:
        return np.zeros((1, 256), dtype=np.uint8)

    n_rows     = max(1, math.ceil(len(data) / 256))
    img_array  = np.zeros((n_rows, 256), dtype=np.uint8)
    for r in range(n_rows):
        img_array[r, :] = _row_entropy(data, r)
    return img_array


def pe_to_entropy_image(filepath: str, img_size: int = 224) -> Image.Image:

    arr = _build_entropy_array(filepath)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img.resize((img_size, img_size), Image.LANCZOS)


def pe_to_entropy_crops(filepath: str, img_size: int = 224) -> list:

    arr    = _build_entropy_array(filepath)
    n_rows = arr.shape[0]

    cut70  = max(1, int(n_rows * 0.70))
    cut10  = max(0, int(n_rows * 0.10))
    cut90  = max(1, int(n_rows * 0.90))

    slices = [
        arr,                     
        arr[:cut70, :],          
        arr[n_rows - cut70:, :], 
        arr[cut10:cut90, :],     
    ]

    crops = []
    for s in slices:
        if s.shape[0] == 0:
            s = arr                 
        img = Image.fromarray(s, mode="L").convert("RGB")
        crops.append(img.resize((img_size, img_size), Image.LANCZOS))
    return crops


def _select_rgb_width(n_bytes: int) -> int:

    if n_bytes < 10_240:       return 32    
    elif n_bytes < 32_768:     return 64   
    elif n_bytes < 65_536:     return 128  
    elif n_bytes < 102_400:    return 256  
    elif n_bytes < 204_800:    return 384   
    elif n_bytes < 512_001:    return 512   
    elif n_bytes < 1_048_576:  return 768   
    else:                      return 1024  


def _build_rgb_array(filepath: str, display_width: int = 256) -> np.ndarray:

    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(data) == 0:
        return np.zeros((1, display_width, 3), dtype=np.uint8)

    width  = display_width
    n_rows = math.ceil(len(data) / width)
    total  = width * n_rows

    if len(data) < total:
        data = np.pad(data, (0, total - len(data)), constant_values=0)
    else:
        data = data[:total]

    grid = data.reshape(n_rows, width)

    R = grid.astype(np.uint8)

    G = np.zeros_like(R)
    for r in range(n_rows):
        G[r, :] = _row_entropy(data, r)

    B = np.where((grid >= 32) & (grid <= 126),
                 np.uint8(255), np.uint8(0)).astype(np.uint8)

    return np.stack([R, G, B], axis=2)


def _rgb_crops_from_array(arr: np.ndarray, img_size: int) -> list:

    n_rows = arr.shape[0]
    cut70  = max(1, int(n_rows * 0.70))
    cut10  = max(0, int(n_rows * 0.10))
    cut90  = max(1, int(n_rows * 0.90))

    slices = [
        arr,
        arr[:cut70, :, :],
        arr[n_rows - cut70:, :, :],
        arr[cut10:cut90, :, :],
    ]
    crops = []
    for s in slices:
        if s.shape[0] == 0:
            s = arr
        img = Image.fromarray(s, mode="RGB")
        crops.append(img.resize((img_size, img_size), Image.LANCZOS))
    return crops


def pe_to_rgb_crops(filepath: str, img_size: int = 224) -> list:

    arr = _build_rgb_array(filepath, display_width=256)
    return _rgb_crops_from_array(arr, img_size)


def pe_to_rgb_crops_mw(filepath: str, img_size: int = 224) -> list:

    n_bytes = os.path.getsize(filepath)
    width   = _select_rgb_width(n_bytes)
    arr     = _build_rgb_array(filepath, display_width=width)
    return _rgb_crops_from_array(arr, img_size)


def _select_width(n_bytes: int) -> int:

    if n_bytes < 10_240:
        return 32
    elif n_bytes < 102_400:
        return 64
    elif n_bytes < 1_048_576:
        return 128
    else:
        return 256


def pe_to_binary_image(filepath: str, img_size: int = 128) -> Image.Image:

    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(data) == 0:
        return Image.new("RGB", (img_size, img_size), 0)

    width  = _select_width(len(data))
    height = math.ceil(len(data) / width)
    total  = width * height

    if len(data) < total:
        data = np.pad(data, (0, total - len(data)), constant_values=0)
    else:
        data = data[:total]

    img = Image.fromarray(data.reshape(height, width), mode="L")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return img.convert("RGB")


def pe_to_rgb_image(filepath: str, display_width: int = 256,
                    max_rows: int = 512) -> Image.Image:

    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(data) == 0:
        return Image.new("RGB", (display_width, 64), (30, 30, 30))

    width  = display_width
    n_rows = math.ceil(len(data) / width)
    total  = width * n_rows

    if len(data) < total:
        data = np.pad(data, (0, total - len(data)), constant_values=0)
    else:
        data = data[:total]

    grid = data.reshape(n_rows, width)


    R = grid.astype(np.uint8)


    G = np.zeros_like(R)
    for r in range(n_rows):
        G[r, :] = _row_entropy(data, r)


    B = np.where((grid >= 32) & (grid <= 126),
                 np.uint8(255), np.uint8(0)).astype(np.uint8)

    if n_rows > max_rows:
        R = R[:max_rows]
        G = G[:max_rows]
        B = B[:max_rows]

    rgb = np.stack([R, G, B], axis=2)
    return Image.fromarray(rgb, mode="RGB")



def generate_visualization_pair(filepath: str):

    img_binary  = pe_to_binary_image(filepath,  img_size=128)
    img_entropy = pe_to_entropy_image(filepath, img_size=224)

    rgb_full   = pe_to_rgb_image(filepath, display_width=256, max_rows=200)
    thumbnail  = rgb_full.resize((200, 200), Image.LANCZOS)

    return img_binary, img_entropy, thumbnail
