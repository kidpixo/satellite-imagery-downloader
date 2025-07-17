#!/usr/bin/env python3
from pathlib import Path
import json
import re
from datetime import datetime
import argparse
import logging
import shutil
import subprocess

import rasterio

from image_downloading import download_image, project_with_scale

# Constants
COORD_REGEX = r'[+-]?\d*\.\d+|d+'

file_dir = Path(__file__).parent
DEFAULT_CONFIG = file_dir / 'preferences.json'
DEFAULT_PREFS = {
    'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    'tile_size': 256,
    'channels': 3,
    'dir': str(file_dir / 'images'),
    'headers': {
        'cache-control': 'max-age=0',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        # this is to coply with the terms of service of the OSM tile server
        'user-agent': 'your_email@example.com - satellite-imagery-downloader for research (https://github.com/yourrepo)'
    },
    # those are example user agents to rotate through requests
    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    ],
    'tl': '',
    'br': '',
    'zoom': '',
    'gdal_translate': {
        'format': 'GTiff',
        'creation_options': []
    },
    'parallel_jobs': 1,
    'request_delay_ms': 200,  # Suggested: 200 ms between requests
    'n_retry': 3  # Default retry count for robust downloading
}

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_config(config_path: Path) -> dict:
    """Load configuration from a JSON file, create default if missing. Gracefully handle malformed JSON."""
    if not config_path.is_file():
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_PREFS, f, indent=2, ensure_ascii=False)
        logging.info(f'Preferences file created in {config_path}')
        return DEFAULT_PREFS.copy()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Malformed JSON in config file '{config_path}': {e}\nPlease fix or delete the file and try again.")
        return None

def save_config(config_path: Path, prefs: dict) -> None:
    """Save configuration to a JSON file."""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)

def validate_config(prefs: dict) -> bool:
    """Check if required fields are present in the config."""
    required = ['url', 'tile_size', 'channels', 'dir', 'headers', 'tl', 'br', 'zoom']
    for key in required:
        if key not in prefs:
            logging.error(f'Missing required config key: {key}')
            return False
    # Check user_agents
    if 'user_agents' not in prefs or not isinstance(prefs['user_agents'], list) or len(prefs['user_agents']) < 3:
        logging.error('Config must include a user_agents list with at least 3 user-agent strings.')
        return False
    return True

def parse_coords(coord_str: str) -> tuple[float, float]:
    """Parse a coordinate string into (lat, lon)."""
    lat, lon = re.findall(COORD_REGEX, coord_str)
    return float(lat), float(lon)

def take_input(messages: list[str]) -> list[str] | None:
    """Prompt user for input, allowing reset or quit."""
    inputs = []
    print('Enter "r" to reset or "q" to exit.')
    for message in messages:
        inp = input(message)
        if inp.lower() == 'q':
            return None
        if inp.lower() == 'r':
            return take_input(messages)
        inputs.append(inp)
    return inputs

def create_world_file(img_path: Path, lat1: float, lon1: float, lat2: float, lon2: float, zoom: int, tile_size: int) -> None:
    """Create a PNG world file (.pgw) for GDAL georeferencing."""
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)
    tl_pixel_x = tl_proj_x * tile_size
    tl_pixel_y = tl_proj_y * tile_size
    br_pixel_x = br_proj_x * tile_size
    br_pixel_y = br_proj_y * tile_size
    width = abs(tl_pixel_x - br_pixel_x)
    height = br_pixel_y - tl_pixel_y
    pixel_size_x = (lon2 - lon1) / width if width != 0 else 0
    pixel_size_y = (lat2 - lat1) / height if height != 0 else 0
    pgw_lines = [
        f"{pixel_size_x}\n",
        "0.0\n",
        "0.0\n",
        f"{-abs(pixel_size_y)}\n",
        f"{lon1}\n",
        f"{lat1}\n"
    ]
    pgw_path = img_path.with_suffix('.pgw')
    with open(pgw_path, 'w') as f:
        f.writelines(pgw_lines)
    logging.info(f'World file saved as {pgw_path.name}')

def build_gdal_command(img_path: Path, prefs: dict) -> list[str]:
    """Build the gdal_translate command from config."""
    gdal_cfg = prefs.get('gdal_translate', {})
    fmt = gdal_cfg.get('format', 'GTiff')
    creation_opts = gdal_cfg.get('creation_options', [])
    output_path = img_path.with_suffix('.tif')
    cmd = [
        'gdal_translate',
        '-of', fmt,
        '-a_srs', 'EPSG:4326',
        str(img_path),
        str(output_path)
    ]
    for opt in creation_opts:
        cmd.extend(['-co', opt])
    return cmd

def run_gdal_translate(img_path: Path, prefs: dict) -> None:
    """Run gdal_translate if available and configured."""
    if shutil.which('gdal_translate') is None:
        logging.warning('gdal_translate not found in PATH. Skipping GeoTIFF conversion.')
        return
    cmd = build_gdal_command(img_path, prefs)
    logging.info(f'Running: {" ".join(cmd)}')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f'gdal_translate output: {result.stdout.strip()}')
        print(f'GeoTIFF saved as {img_path.with_suffix(".tif")}')
    except subprocess.CalledProcessError as e:
        logging.error(f'gdal_translate failed: {e.stderr.strip()}')

def clean_dir(img_dir: Path) -> None:
    """Delete all files in the image directory."""
    if img_dir.is_dir():
        for file in img_dir.iterdir():
            if file.is_file():
                file.unlink()
        logging.info(f'Cleaned directory: {img_dir}')
    else:
        logging.warning(f'Directory does not exist: {img_dir}')

def log_config_info(prefs: dict) -> None:
    """Log the config URL as info and all other config (except headers and user_agents) as debug."""
    url = prefs.get('url', None)
    if url:
        logging.info(f'Config URL: {url}')
    for k, v in prefs.items():
        if k in ('headers', 'url', 'user_agents'):
            continue
        logging.debug(f'Config {k}: {v}')
    if 'user_agents' in prefs:
        logging.debug(f'user_agents: {prefs["user_agents"]}')

def write_mosaic(mosaic, output_path, transform, crs, fmt, creation_opts, dtype=None):
    import rasterio
    meta = {
        'driver': fmt,
        'height': mosaic.shape[1] if mosaic.ndim == 3 else mosaic.shape[0],
        'width': mosaic.shape[2] if mosaic.ndim == 3 else mosaic.shape[1],
        'count': mosaic.shape[0] if mosaic.ndim == 3 else 1,
        'dtype': dtype if dtype is not None else mosaic.dtype,
        'transform': transform,
        'crs': crs
    }
    # Add creation options
    for opt in creation_opts:
        if '=' in opt:
            k, v = opt.split('=', 1)
            meta[k.lower()] = v
    with rasterio.open(output_path, 'w', **meta) as dst:
        if mosaic.ndim == 3:
            dst.write(mosaic)
        else:
            dst.write(mosaic, 1)

def main(config_path: Path, debug: bool = False) -> None:
    logging.info(f"Using config file: {config_path}")
    prefs = load_config(config_path)
    log_config_info(prefs)
    if not validate_config(prefs):
        return
    img_dir = Path(prefs['dir'])
    img_dir.mkdir(parents=True, exist_ok=True)
    if (prefs['tl'] == '') or (prefs['br'] == '') or (prefs['zoom'] == ''):
        messages = ['Top-left corner: ', 'Bottom-right corner: ', 'Zoom level: ']
        inputs = take_input(messages)
        if inputs is None:
            return
        prefs['tl'], prefs['br'], prefs['zoom'] = inputs
        save_config(config_path, prefs)
    lat1, lon1 = parse_coords(prefs['tl'])
    lat2, lon2 = parse_coords(prefs['br'])
    zoom = int(prefs['zoom'])
    channels = int(prefs['channels'])
    tile_size = int(prefs['tile_size'])
    user_agents = prefs.get('user_agents', None)
    n_retry = int(prefs.get('n_retry', 3))
    mosaic, out_transform, out_meta = download_image(
        lat1, lon1, lat2, lon2, zoom, prefs['url'], prefs['headers'], tile_size, channels,
        user_agents=user_agents, debug=debug, debug_dir=img_dir,
        request_delay_ms=prefs.get('request_delay_ms'), n_retry=n_retry)
    if mosaic is None or mosaic.size == 0:
        logging.error("The downloaded image is empty. Please check your coordinates, zoom level, and network connection.")
        return
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f'img_{timestamp}.tif'
    img_path = img_dir / name
    gdal_cfg = prefs.get('gdal_translate', {})
    fmt = gdal_cfg.get('format', 'GTiff')
    creation_opts = gdal_cfg.get('creation_options', [])
    write_mosaic(mosaic, img_path, out_transform, out_meta.get('crs', None), fmt, creation_opts, dtype=mosaic.dtype)
    logging.info(f'Saved as {name}')
    # Optionally, create a PNG for preview (not georeferenced)
    # png_path = img_path.with_suffix('.png')
    # from PIL import Image
    # arr = np.moveaxis(mosaic, 0, -1)  # (bands, h, w) -> (h, w, bands)
    # Image.fromarray(arr).save(png_path)
    # logging.info(f'Preview PNG saved as {png_path.name}')
    # create_world_file and gdal_translate are not needed if using rasterio for GeoTIFF
    # If you want to keep world file or gdal_translate, adapt as needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Imagery Downloader")
    parser.add_argument('-c', '--config', type=str, default=str(DEFAULT_CONFIG), help='Path to config file (default: preferences.json)')
    parser.add_argument('-x', '--clean', action='store_true', help='Clean the image directory specified in the config and exit')
    parser.add_argument('-r', '--run', action='store_true', help='Run the downloader pipeline')
    parser.add_argument('-g', '--gdal', action='store_true', help='Only apply the gdal_translate transformation to the latest PNG in the image directory')
    parser.add_argument('-l', '--loglevel', type=str, default='INFO', help='Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO')
    parser.add_argument('-d', '--debug', action='store_true', help='Store all single tiles as PNGs for debugging')
    parser.add_argument('-m', '--mosaic-only', action='store_true', help='Only mosaic all GeoTIFF tiles in the image directory and save the result')
    parser.add_argument('--check-tiles', action='store_true', help='Check for missing tiles in the grid and output the number and filenames of missing tiles')
    args = parser.parse_args()
    logging.getLogger().setLevel(args.loglevel.upper())
    if not (args.clean or args.run or args.gdal or args.mosaic_only or args.check_tiles):
        parser.print_help()
    else:
        config_path = Path(args.config)
        prefs = load_config(config_path)
        img_dir = Path(prefs['dir'])
        if args.clean:
            clean_dir(img_dir)
        if args.run:
            main(config_path, debug=args.debug)
        if args.gdal:
            # Find the latest PNG in the image directory
            pngs = sorted(img_dir.glob('img_*.png'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not pngs:
                logging.error('No PNG images found in the image directory.')
            else:
                latest_png = pngs[0]
                logging.info(f'Applying gdal_translate to {latest_png}')
                run_gdal_translate(latest_png, prefs)
        if args.check_tiles:
            import re
            tile_pattern = re.compile(r"tile_z(\d+)_x(\d+)_y(\d+)\.png")
            # Find all existing tile filenames in the root image directory
            existing_tiles = set()
            for f in img_dir.glob("tile_z*_x*_y*.png"):
                if f.is_file() and f.parent == img_dir:
                    m = tile_pattern.match(f.name)
                    if m:
                        z, x, y = map(int, m.groups())
                        existing_tiles.add((z, x, y))
            # Always compute expected grid as full rectangle from min/max X,Y
            if not existing_tiles:
                logging.error('No tiles found to check.')
                print("No tiles found to check.")
            else:
                zs = set(z for z, x, y in existing_tiles)
                xs = set(x for z, x, y in existing_tiles)
                ys = set(y for z, x, y in existing_tiles)
                z = list(zs)[0] if zs else 0
                expected_tiles = set((z, x, y) for x in range(min(xs), max(xs)+1) for y in range(min(ys), max(ys)+1))
                missing = expected_tiles - existing_tiles
                logging.info(f"Total expected tiles: {len(expected_tiles)}")
                logging.info(f"Total existing tiles: {len(existing_tiles)}")
                logging.info(f"Missing tiles: {len(missing)}")
                # Append missing tiles to CSV as failed
                csv_path = img_dir / 'tiles_debug.csv'
                if missing:
                    if csv_path.exists():
                        import csv
                        with open(csv_path, newline='') as csvfile:
                            rows = list(csv.DictReader(csvfile))
                        existing_filenames = set(row['tile_filename'] for row in rows)
                        url_lookup = {row['tile_filename']: row.get('url', '') for row in rows}
                    else:
                        rows = []
                        existing_filenames = set()
                        url_lookup = {}
                    new_rows = []
                    for z, x, y in sorted(missing):
                        fname = f"tile_z{z}_x{x}_y{y}.png"
                        if fname not in existing_filenames:
                            # Generate URL from config if not found in url_lookup
                            url = url_lookup.get(fname, '')
                            if not url:
                                url_template = prefs.get('url', '')
                                url = url_template.format(z=z, x=x, y=y)
                            new_rows.append({'success': 'False', 'tile_filename': fname, 'url': url})
                    if new_rows:
                        rows.extend(new_rows)
                        with open(csv_path, 'w', newline='') as csvfile:
                            import csv
                            writer = csv.DictWriter(csvfile, fieldnames=['success', 'tile_filename', 'url'])
                            writer.writeheader()
                            writer.writerows(rows)
                        logging.info(f"Added {len(new_rows)} missing tiles to CSV as failed.")
                    for z, x, y in sorted(missing):
                        logging.debug(f"Missing tile: tile_z{z}_x{x}_y{y}.png")
                else:
                    print("No missing tiles.")
        if args.mosaic_only:
            import re
            import numpy as np
            from datetime import datetime
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import rasterio
            from rasterio.transform import from_origin
            import csv
            import requests
            from tqdm import tqdm
            tile_pattern = re.compile(r"tile_z(\d+)_x(\d+)_y(\d+)\.png")
            # Step 1: If 'corrupt' subfolder exists and contains images, mark those as success=False in the CSV
            corrupt_dir = img_dir / 'corrupt'
            csv_path = img_dir / 'tiles_debug.csv'
            corrupt_files = []
            if corrupt_dir.exists() and corrupt_dir.is_dir():
                corrupt_files = [f.name for f in corrupt_dir.iterdir() if f.is_file() and tile_pattern.match(f.name)]
                logging.info(f"Found {len(corrupt_files)} files in 'corrupt' directory.")
            else:
                logging.info("No 'corrupt' directory found or it is empty.")
            if corrupt_files and csv_path.exists():
                # Load CSV, set success=False for corrupt files
                with open(csv_path, newline='') as csvfile:
                    rows = list(csv.DictReader(csvfile))
                updated = False
                for row in rows:
                    if row['tile_filename'] in corrupt_files:
                        if row['success'].strip().lower() != 'false':
                            row['success'] = 'False'
                            updated = True
                            logging.debug(f"Marked {row['tile_filename']} as success=False in CSV.")
                if updated:
                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=['success', 'tile_filename', 'url'])
                        writer.writeheader()
                        writer.writerows(rows)
                    logging.info(f"Updated CSV: marked {len(corrupt_files)} corrupt images as success=False.")
                else:
                    logging.info("No updates needed in CSV for corrupt files.")
            # Step 2: Redownload failed tiles if tiles_debug.csv is present
            if csv_path.exists():
                with open(csv_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
                    failed = [row for row in rows if row['success'].strip().lower() == 'false']
                if failed:
                    logging.info(f"Redownloading {len(failed)} failed tiles from tiles_debug.csv...")
                    headers = prefs.get('headers', {})
                    for row in tqdm(failed, desc="Redownloading failed tiles"):
                        url = row['url']
                        tile_filename = row['tile_filename']
                        out_path = img_dir / tile_filename
                        try:
                            r = requests.get(url, headers=headers, timeout=30)
                            r.raise_for_status()
                            with open(out_path, 'wb') as f:
                                f.write(r.content)
                            row['success'] = 'True'
                            logging.info(f"Redownloaded {tile_filename} successfully.")
                        except Exception as e:
                            logging.error(f"Failed to redownload {tile_filename}: {e}")
                    # Rewrite CSV with updated success values
                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=['success', 'tile_filename', 'url'])
                        writer.writeheader()
                        writer.writerows(rows)
                    logging.info("CSV updated after redownload attempts.")
                else:
                    logging.info("No failed tiles to redownload in CSV.")
            else:
                logging.info("No tiles_debug.csv found; skipping redownload step.")
            # Step 3: Mosaic only images in the root image directory (not in subfolders)
            gdal_cfg = prefs.get('gdal_translate', {})
            fmt = gdal_cfg.get('format', 'GTiff')
            creation_opts = gdal_cfg.get('creation_options', [])
            ext = fmt.lower() if fmt.lower() != 'gtiff' else 'tif'
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            mosaic_name = f'img_{timestamp}.{ext}'
            output = img_dir / mosaic_name
            # Only use PNGs in the root of the image directory
            tiles = []
            for f in img_dir.glob("tile_z*_x*_y*.png"):
                if f.is_file() and f.parent == img_dir:
                    m = tile_pattern.match(f.name)
                    if m:
                        z, x, y = map(int, m.groups())
                        tiles.append((x, y, f))
            logging.info(f"Found {len(tiles)} tiles in image directory root for mosaicing.")
            if not tiles:
                logging.error(f"No PNG tile files found in {img_dir} (root only, not subfolders). Mosaic step skipped.")
            else:
                xs = sorted(set(x for x, y, _ in tiles))
                ys = sorted(set(y for x, y, _ in tiles))
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                grid = {(x, y): f for x, y, f in tiles}
                tile_size = int(prefs.get('tile_size', 256))
                width = (x_max - x_min + 1) * tile_size
                height = (y_max - y_min + 1) * tile_size
                def load_tile(x, y):
                    tile_file = grid.get((x, y))
                    if tile_file:
                        try:
                            with rasterio.open(tile_file) as src:
                                arr = src.read()
                                # rasterio returns (bands, h, w), convert to (h, w, bands)
                                arr = arr.transpose(1, 2, 0)
                                if arr.shape[2] > 3:
                                    arr = arr[:, :, :3]
                                elif arr.shape[2] < 3:
                                    # pad to 3 channels if needed
                                    arr = np.pad(arr, ((0,0),(0,0),(0,3-arr.shape[2])), mode='constant')
                                return (x, y, arr.astype(np.uint8))
                        except Exception as e:
                            logging.error(f"Unreadable tile at ({x},{y}): {tile_file.name} - {e}")
                            return (x, y, np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
                    else:
                        logging.warning(f"Missing tile at ({x},{y}) - filling with black.")
                        return (x, y, np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
                mosaic_array = np.zeros((height, width, 3), dtype=np.uint8)
                futures = []
                with ThreadPoolExecutor() as executor:
                    for x in xs:
                        for y in ys:
                            futures.append(executor.submit(load_tile, x, y))
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Mosaicing tiles"):
                        x, y, tile_img = f.result()
                        px = (x - x_min) * tile_size
                        py = (y - y_min) * tile_size
                        mosaic_array[py:py+tile_size, px:px+tile_size, :] = tile_img
                logging.info(f"Writing mosaic to {output} with driver={fmt}...")
                # Compute georeferencing from tile grid
                from image_downloading import tile_xy_to_bounds
                left, _, _, top = tile_xy_to_bounds(x_min, y_min, z, tile_size)
                _, bottom, right, _ = tile_xy_to_bounds(x_max, y_max, z, tile_size)
                from affine import Affine
                pixel_width = (right - left) / width
                pixel_height = (top - bottom) / height
                transform = Affine(pixel_width, 0, left, 0, -abs(pixel_height), top)
                # Write mosaic using shared function
                write_mosaic(
                    mosaic_array.transpose(2, 0, 1),  # (bands, height, width)
                    output,
                    transform,
                    'EPSG:3857',
                    fmt,
                    creation_opts,
                    dtype=mosaic_array.dtype
                )
                logging.info(f"Mosaic saved as {output}")
