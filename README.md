# Satellite Imagery Downloader

## Problem-First Approach

**Problem:**
Developers, GIS analysts, and researchers often need to extract high-resolution satellite or map imagery for custom regions, automate bulk downloads, or prepare georeferenced images for further analysis in GIS tools (like QGIS) or web mapping frameworks (like Leaflet.js). Manual downloading is tedious, error-prone, and often not reproducible.

**Solution:**
This tool automates the process of downloading, assembling, and georeferencing map tiles from various providers (Google, OSM, Esri, etc.), saving them as PNGs, and optionally converting them to GIS-ready Cloud Optimized GeoTIFFs (COGs) or other formats using GDAL. It supports parallel downloads, flexible configuration, and is designed for integration into modern geospatial workflows.

---

## Story (Why): Purpose, Motivation, and Design

- **Purpose:**
  - Download and assemble map/satellite tiles for any rectangular region, at any zoom, from any provider using a URL template.
  - Save as PNG and (optionally) convert to COG/GeoTIFF with full georeferencing for use in GIS or web mapping.
- **Motivation:**
  - Reproducible, scriptable, and automatable extraction of imagery for analysis, visualization, or machine learning.
- **Design Choices:**
  - Configuration-driven (JSON), CLI-first, extensible, and developer-friendly.
  - Supports parallel downloads (joblib), progress bars (tqdm), and GDAL integration for advanced raster output.

---

## Quickstart (Code: How)

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Tool
```bash
python main.py -r
```
- On first run, a `preferences.json` config file is created. Edit it or rerun to enter coordinates interactively.

### 3. Download and Convert in One Go
```bash
python main.py -r -g
```
- Downloads imagery and immediately applies the GDAL transformation (e.g., to COG/GeoTIFF).

### 4. Clean Output Directory
```bash
python main.py -x
```

### 5. Only Apply GDAL Transformation to Latest Image
```bash
python main.py -g
```

### 6. Set Log Level
```bash
python main.py -r -l DEBUG
```

### 7. Parallel Downloads
- Set `"parallel_jobs": 4` (or any integer >1) in your `preferences.json` to enable parallel tile downloads.

---

## Configuration (Context: Where/When)

Edit `preferences.json` to control all aspects of the workflow:

```json
{
  "url": "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
  "tile_size": 256,
  "channels": 3,
  "dir": "./images",
  "headers": { ... },
  "tl": "41.3703, 14.3262",
  "br": "41.3186, 14.4050",
  "zoom": "16",
  "parallel_jobs": 4,
  "request_delay_ms": 200,
  "n_retry": 3,
  "gdal_translate": {
    "format": "COG",
    "creation_options": ["COMPRESS=JPEG", "TILING_SCHEME=GoogleMapsCompatible"]
  }
}
```

- **url**: Tile URL template (see below for examples)
- **tile_size**: Size of each tile in pixels
- **channels**: 3 (RGB) or 4 (RGBA)
- **dir**: Output directory
- **headers**: HTTP headers for requests
- **tl/br**: Top-left and bottom-right coordinates (lat, lon)
- **zoom**: Zoom level
- **parallel_jobs**: Number of parallel download jobs (optional)
- **request_delay_ms**: Delay (in ms) between requests (with random jitter for robustness)
- **n_retry**: Number of retry attempts for each tile (robust download)
- **gdal_translate**: GDAL output settings (format, creation options)

#### Tile URL Examples
- Google Satellite: `https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}`
- OpenStreetMap: `https://tile.openstreetmap.org/{z}/{x}/{y}.png` ( be careful with OSM server!!)
- Esri: `https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`

---

## Usage Patterns & Integration

- **GIS Workflow:** Output COG/GeoTIFF can be loaded directly into QGIS, ArcGIS, or any modern GIS.
- **Web Mapping:** PNG or COG output can be served as a tile layer in Leaflet.js, MapLibre, or similar.
- **Automation:** Integrate into scripts, CI/CD, or data pipelines for reproducible geospatial analysis.

---

## Advanced Features

- **Parallel Downloads:**
  - Set `parallel_jobs` in config for faster downloads (uses joblib if available).
- **Robust Retry & Jitter:**
  - Set `n_retry` in config to control how many times each tile is retried on failure (network or decode errors).
  - Set `request_delay_ms` to add a delay (with random jitter) between requests, reducing the risk of being blocked and avoiding synchronized requests in parallel jobs.
- **GDAL Integration:**
  - Use `gdal_translate` settings in config to produce COG, GeoTIFF, or other formats with custom options.
- **Flexible CLI:**
  - `-r/--run` to download, `-g/--gdal` to convert, `-x/--clean` to clean, `-l/--loglevel` to set verbosity.
  - Combine flags for custom workflows.
- **Interactive or Config-Driven:**
  - Enter coordinates interactively or set them in the config for automation.

---

## Troubleshooting & Failure Scenarios

- **No Tiles Downloaded / Empty Image:**
  - Check coordinates, zoom, and network. The script will log an error if the image is empty.
- **GDAL Not Found:**
  - Ensure `gdal_translate` is installed and in your PATH.
- **Permission Errors:**
  - Make sure the output directory is writable.
- **Parallel Download Issues:**
  - If you see errors with parallel jobs, try reducing `parallel_jobs` or set to 1 to disable.
- **HTTP Errors / Blocked Requests:**
  - Some providers may block automated requests. Try adjusting headers, using a different provider, or increasing `request_delay_ms` and `n_retry` for more robust and polite downloading.
- **Corrupted or Missing Tiles:**
  - The downloader will retry up to `n_retry` times and add random jitter to delays. If a tile still fails, it will be skipped with a warning in the logs.

---

## Example Workflows

### Download, Convert, and Load in QGIS
```bash
python main.py -r -g -l INFO
# Then open the resulting .tif in QGIS
```

### Clean Output, Download, and Convert
```bash
python main.py -x -r -g
```

### Use in a Web Map
- Serve the PNG or COG output as a static tile in Leaflet.js or MapLibre.

### Download, Check, Redownload, and Mosaic (Robust Workflow)

1. **Download all tiles and create initial mosaic:**
   ```bash
   python main.py -r
   ```
   - Downloads all tiles and creates a georeferenced mosaic.

2. **(Optional) Manually check for corrupted or missing tiles:**
   - Open the `images/` directory and inspect the PNG tiles.
   - Move any corrupted or incomplete tiles to the `images/corrupt/` subfolder (create it if it doesn't exist).

3. **Mark corrupt/missing tiles and update CSV:**
   ```bash
   python main.py --check-tiles
   ```
   - Updates `tiles_debug.csv` to mark corrupt/missing tiles as failed and generates download URLs for them.

4. **Redownload only failed/missing tiles and create a new mosaic:**
   ```bash
   python main.py --mosaic-only
   ```
   - Redownloads only the failed/missing tiles, updates the CSV, and creates a new georeferenced mosaic from all available tiles.

---

## Contributing & Extending
- Fork, open issues, or submit PRs for new features (e.g., new providers, output formats, async support).
- Consider integrating with cloud storage, web APIs, or other geospatial tools.

---

## License & Attribution
- Respect the terms of use of each map provider.
- Example images © Google, Esri, OpenStreetMap, as noted.

---

## Visual Examples
![](img/img_2.png)
<nobr><sup><sup>© 2022 Google</sup></sup></nobr>

![](img/img_3.png)
<nobr><sup><sup>© 2022 Google</sup></sup></nobr>

![](img/img_4.png)
<nobr><sup><sup>© 2022 Google</sup></sup></nobr>

![](img/img_5.png)
<nobr><sup><sup>© 2023 Google</sup></sup></nobr>

![](img/img_6.png)
<nobr><sup><sup>© OpenStreetMap</sup></sup></nobr>
