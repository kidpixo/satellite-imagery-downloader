import requests
import numpy as np
import threading
from tqdm import tqdm
import time
import random
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from PIL import Image  # Ensure PIL.Image is always available
import io  # Ensure io is always available


def download_tile(url, headers, user_agents=None, max_retries=3, min_delay=0.5, max_delay=2.0):
    """
    Download a single tile as a rasterio dataset with retry logic, random delay, and optional user-agent rotation.
    Returns a rasterio dataset (in-memory) or None on failure.
    Handles PNG tiles (e.g., OSM) using PIL for compatibility.
    """
    attempt = 0
    last_exception = None
    while attempt < max_retries:
        try:
            if user_agents:
                headers = headers.copy()
                headers['user-agent'] = random.choice(user_agents)
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                tile_bytes = response.content
                # Always try rasterio, but do not call .read() here
                try:
                    memfile = MemoryFile(tile_bytes)
                    return memfile.open()
                except Exception:
                    # If rasterio.open fails, fallback to PIL
                    img = Image.open(io.BytesIO(tile_bytes)).convert('RGB')
                    arr = np.array(img)
                    arr = np.moveaxis(arr, -1, 0)  # (H, W, 3) -> (3, H, W)
                    memfile = MemoryFile()
                    meta = {
                        'driver': 'GTiff',
                        'dtype': arr.dtype,
                        'count': arr.shape[0],
                        'height': arr.shape[1],
                        'width': arr.shape[2],
                        'crs': None,
                        'transform': None
                    }
                    with memfile.open(**meta) as dataset:
                        dataset.write(arr)
                    return memfile.open()
            else:
                last_exception = Exception(f"HTTP {response.status_code}")
        except Exception as e:
            last_exception = e
        attempt += 1
        if attempt < max_retries:
            # Add random jitter to delay
            jitter = random.uniform(0, 0.5)
            time.sleep(random.uniform(min_delay, max_delay) + jitter)
    return None


# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def tile_xy_to_bounds(x, y, z, tile_size=256):
    """
    Returns the bounds (left, bottom, right, top) of a tile in Web Mercator (EPSG:3857).
    """
    import math
    n = 2 ** z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    # Convert to Web Mercator meters
    def lonlat_to_mercator(lon, lat):
        R = 6378137.0
        x = R * math.radians(lon)
        y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return x, y
    left, top = lonlat_to_mercator(lon_left, lat_top)
    right, bottom = lonlat_to_mercator(lon_right, lat_bottom)
    return left, bottom, right, top

def tile_xy_to_transform(x, y, z, tile_size=256):
    """
    Returns the affine transform for a tile in Web Mercator (EPSG:3857).
    """
    left, bottom, right, top = tile_xy_to_bounds(x, y, z, tile_size)
    pixel_width = (right - left) / tile_size
    pixel_height = (top - bottom) / tile_size
    from affine import Affine
    return Affine(pixel_width, 0, left, 0, -pixel_height, top)


def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3, user_agents=None, debug=False, debug_dir=None, request_delay_ms=None, n_retry=3):
    """
    Downloads a map region. Returns a tuple (mosaic, out_transform, out_meta):
    - mosaic: numpy array (bands, height, width)
    - out_transform: affine transform for the mosaic
    - out_meta: rasterio metadata dict
    """
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)
    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)
    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    tile_datasets = []
    total_tiles = (br_tile_y - tl_tile_y + 1) * (br_tile_x - tl_tile_x + 1)
    pbar = tqdm(total=total_tiles, desc="Downloading tiles", unit="tile")

    tile_csv_rows = []  # For debug: collect [success, tile_filename, url]
    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile_url = url.format(x=tile_x, y=tile_y, z=zoom)
            max_tile_retries = n_retry
            tile_attempt = 0
            tile_filename = f"tile_z{zoom}_x{tile_x}_y{tile_y}.png"
            success = False
            arr = None
            while tile_attempt < max_tile_retries:
                ds = download_tile(tile_url, headers, user_agents=user_agents, max_retries=n_retry)
                if ds is not None:
                    try:
                        arr = ds.read()
                        success = True
                    except Exception:
                        ds.close()
                        response = requests.get(tile_url, headers=headers, timeout=10)
                        img = Image.open(io.BytesIO(response.content)).convert('RGB')
                        arr = np.array(img)
                        arr = np.moveaxis(arr, -1, 0)
                        success = True
                    break  # Success, exit retry loop
                else:
                    tile_attempt += 1
                    if tile_attempt < max_tile_retries:
                        jitter = random.uniform(0, 0.5)
                        time.sleep(2.0 + jitter)
                    else:
                        import logging
                        logging.warning(f"Giving up on tile at x={tile_x}, y={tile_y} after {max_tile_retries} attempts (download failed).")
                        break
            # Save debug PNG if requested
            if debug and debug_dir is not None and arr is not None:
                arr_png = np.moveaxis(arr, 0, -1)
                img = Image.fromarray(arr_png)
                fname = debug_dir / tile_filename
                img.save(fname)
            # Record CSV row for this tile
            if debug and debug_dir is not None:
                tile_csv_rows.append([str(success), tile_filename, tile_url])
            # Only add to mosaic if successful
            if success and arr is not None:
                from affine import Affine
                transform = tile_xy_to_transform(tile_x, tile_y, zoom, tile_size)
                crs = 'EPSG:3857'
                memfile2 = rasterio.io.MemoryFile()
                meta = {
                    'driver': 'GTiff',
                    'dtype': arr.dtype,
                    'count': arr.shape[0],
                    'height': arr.shape[1],
                    'width': arr.shape[2],
                    'crs': crs,
                    'transform': transform
                }
                with memfile2.open(**meta) as flipped_ds:
                    flipped_ds.write(arr)
                tile_datasets.append(memfile2.open())
            pbar.update(1)

    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        thread = threading.Thread(target=build_row, args=[tile_y])
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    pbar.close()

    if not tile_datasets:
        return None, None, None

    # Mosaic all tiles
    mosaic, _ = merge(tile_datasets)
    # Compute the bounds of the full tile grid in EPSG:3857
    left, _, _, top = tile_xy_to_bounds(tl_tile_x, tl_tile_y, zoom, tile_size)
    _, bottom, right, _ = tile_xy_to_bounds(br_tile_x, br_tile_y, zoom, tile_size)
    width = mosaic.shape[2]
    height = mosaic.shape[1]
    pixel_width = (right - left) / width
    pixel_height = (top - bottom) / height
    from affine import Affine
    transform = Affine(pixel_width, 0, left, 0, -abs(pixel_height), top)
    print(f"DEBUG: tile grid bounds EPSG:3857: left={left}, right={right}, top={top}, bottom={bottom}")
    print(f"DEBUG: mosaic size: width={width}, height={height}")
    print(f"DEBUG: pixel_width={pixel_width}, pixel_height={pixel_height}")
    print(f"DEBUG: transform={transform}")
    out_meta = tile_datasets[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "crs": "EPSG:3857"
    })
    # Close all datasets
    for ds in tile_datasets:
        ds.close()
    # After all threads join, if debug, write CSV
    if debug and tile_csv_rows:
        import csv
        csv_path = debug_dir / 'tiles_debug.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['success', 'tile_filename', 'url'])
            writer.writerows(tile_csv_rows)
        import logging
        logging.info(f"CSV with tile URLs saved as {csv_path}")
    return mosaic, transform, out_meta


def image_size(lat1: float, lon1: float, lat2: float, lon2: float, zoom: int, tile_size: int = 256):
    """ Calculates the size of an image without downloading it. Returns the width and height in pixels as a tuple. """
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)
    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)
    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
