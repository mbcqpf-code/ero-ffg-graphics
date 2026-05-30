import matplotlib
matplotlib.use('Agg') # CRITICAL for headless GitHub Actions servers
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import xarray as xr
import requests
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import scipy.ndimage as ndimage
import cfgrib

warnings.filterwarnings('ignore')

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def get_fxx_range(cycle, target_day):
    """Returns the forecast hours and the completion status for Day 1/Day 2 (HREF)."""
    if target_day == 1:
        if cycle == 0: return range(12, 37), "Full"
        elif cycle == 6: return range(6, 31), "Full"
        elif cycle == 12: return range(0, 25), "Full"
        elif cycle == 18: return range(0, 19), "Full"
    elif target_day == 2:
        if cycle == 0: return range(36, 49), "Partial (12hr)"
        elif cycle == 6: return range(30, 49), "Partial (18hr)"
        elif cycle == 12: return range(24, 49), "Full"
        elif cycle == 18: return range(18, 43), "Full"
    return None, None

def get_latest_href_run(target_day):
    now = datetime.utcnow()
    current_cycle_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"
    headers = {"User-Agent": "Mozilla/5.0"}

    for i in range(6):
        dt = current_cycle_time - timedelta(hours=6 * i)
        cycle = dt.hour
        date_str = dt.strftime("%Y%m%d")

        fxx_range, status = get_fxx_range(cycle, target_day)
        if fxx_range is None: continue

        last_fxx = fxx_range[-1]
        file_name = f"href.t{cycle:02d}z.conus.ffri.f{last_fxx:02d}.grib2"
        dir_path = f"/href.{date_str}/ensprod"
        params = {"dir": dir_path, "file": file_name, "var_PPFFG": "on", "all_lev": "on"}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"✅ Locked in fully uploaded HREF run: Date={date_str}, Cycle={cycle:02d}z")
                return date_str, cycle, fxx_range, status
        except: pass

    raise ValueError(f"Could not find fully uploaded HREF runs for Day {target_day}.")

def get_rrfs_fxx_range(cycle, target_day):
    """REFS runs out to 60 hours, so it ALWAYS has a full Day 2!"""
    if target_day == 1:
        if cycle == 0: return range(12, 37), "Full"
        elif cycle == 6: return range(6, 31), "Full"
        elif cycle == 12: return range(0, 25), "Full"
        elif cycle == 18: return range(0, 19), "Full"
    elif target_day == 2:
        if cycle == 0: return range(36, 61), "Full"
        elif cycle == 6: return range(30, 55), "Full"
        elif cycle == 12: return range(24, 49), "Full"
        elif cycle == 18: return range(18, 43), "Full"
    return None, None

def get_latest_rrfs_run(target_day):
    now = datetime.utcnow()
    current_cycle_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    base_url = "https://noaa-rrfs-pds.s3.amazonaws.com"

    for i in range(6):
        dt = current_cycle_time - timedelta(hours=6 * i)
        cycle = dt.hour
        date_str = dt.strftime("%Y%m%d")

        fxx_range, status = get_rrfs_fxx_range(cycle, target_day)
        if fxx_range is None: continue
        last_fxx = fxx_range[-1]

        folder_path = f"/rrfs_public/refs.{date_str}/{cycle:02d}/enspost"
        file_name = f"refs.t{cycle:02d}z.ffri.f{last_fxx:02d}.conus.grib2"
        idx_url = f"{base_url}{folder_path}/{file_name}.idx"

        try:
            if requests.head(idx_url, timeout=5).status_code == 200:
                print(f"✅ Locked in fully uploaded REFS run: Date={date_str}, Cycle={cycle:02d}z")
                return date_str, cycle, fxx_range, folder_path, base_url, status
        except: pass

    raise ValueError(f"Could not find fully uploaded REFS runs.")

def download_aws_subset(grib_url, idx_url, search_str, local_file):
    idx_resp = requests.get(idx_url, timeout=10)
    if idx_resp.status_code != 200: return False
    lines = idx_resp.text.strip().split('\n')
    starts, ends = [], []
    for i, line in enumerate(lines):
        if search_str in line:
            parts = line.split(':')
            starts.append(int(parts[1]))
            if i + 1 < len(lines): ends.append(int(lines[i+1].split(':')[1]) - 1)
            else: ends.append(None)
    if not starts: return False
    min_byte = min(starts)
    max_byte = max([e for e in ends if e is not None]) if None not in ends else ""
    headers = {"Range": f"bytes={min_byte}-{max_byte}"}
    grib_resp = requests.get(grib_url, headers=headers, timeout=30)
    if grib_resp.status_code in (200, 206):
        with open(local_file, 'wb') as f: f.write(grib_resp.content)
        return True
    return False

def apply_heavy_smoothing(grid, sigma=5.0):
    if grid is None: return None
    grid_filled = np.nan_to_num(grid, nan=0.0)
    smoothed_grid = ndimage.gaussian_filter(grid_filled, sigma=sigma)
    return smoothed_grid

# ==========================================
# 2. DATA PROCESSING (HREF & REFS)
# ==========================================

href_results = {}
refs_results = {}

# Set up directories
href_download_dir = Path("href_downloads")
href_download_dir.mkdir(exist_ok=True)

refs_download_dir = Path("refs_downloads")
refs_download_dir.mkdir(exist_ok=True)

for target_day in [1, 2]:
    # --- PROCESS HREF ---
    print(f"\n--- PROCESSING HREF DAY {target_day} ---")
    try:
        today_date, cycle, fxx_range, status = get_latest_href_run(target_day)
        
        max_ffg_grid = None
        lats = None
        lons = None
        ero_start_fxx = fxx_range[0]

        for fxx in fxx_range:
            file_name = f"href.t{cycle:02d}z.conus.ffri.f{fxx:02d}.grib2"
            dir_path = f"/href.{today_date}/ensprod"
            url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"
            params = {"dir": dir_path, "file": file_name, "var_PPFFG": "on", "all_lev": "on"}
            headers = {"User-Agent": "Mozilla/5.0"}
            local_file = href_download_dir / f"{file_name}_subset"

            download_success = False
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, headers=headers, timeout=30)
                    if response.status_code == 200 and len(response.content) > 1000:
                        with open(local_file, 'wb') as f: f.write(response.content)
                        download_success = True
                        break
                    else:
                        time.sleep(5) 
                except:
                    time.sleep(5)

            if not download_success:
                print(f"  -> Skipping f{fxx:02d} (Download failed)")
                continue

            for idx_file in href_download_dir.glob('*.idx'):
                try: idx_file.unlink()
                except: pass

            try:
                hour_max = None
                for N in [1, 3, 6]:
                    start_hour = fxx - N
                    if start_hour < ero_start_fxx: continue

                    target_step = f"{start_hour}-{fxx}"
                    try:
                        ds_interval = xr.open_dataset(local_file, engine="cfgrib",
                                                      backend_kwargs={'filter_by_keys': {'stepRange': target_step}})
                        temp_data = ds_interval.ppffg.values
                        if hour_max is None: hour_max = temp_data
                        else: hour_max = np.maximum(hour_max, temp_data)

                        if lats is None:
                            lats = ds_interval.latitude.values
                            lons = ds_interval.longitude.values
                        ds_interval.close()
                    except: continue

                if hour_max is not None:
                    if max_ffg_grid is None: max_ffg_grid = hour_max
                    else: max_ffg_grid = np.maximum(max_ffg_grid, hour_max)
                    print(f"  -> Processed f{fxx:02d}")

            except Exception as e:
                print(f"  -> Error on f{fxx:02d}: {e}")

            time.sleep(1)

        href_results[target_day] = {
            'max': max_ffg_grid, 'lats': lats, 'lons': lons,
            'cycle': cycle, 'status': status
        }
    except ValueError as e:
        print(f"  -> {e} Skipping Day {target_day}...")

    # --- PROCESS REFS ---
    print(f"\n--- PROCESSING REFS DAY {target_day} ---")
    try:
        date_str, cycle, fxx_range, folder_path, base_url, status = get_latest_rrfs_run(target_day)

        max_ffg_grid = None
        lats = None
        lons = None
        ero_start_fxx = fxx_range[0]

        for fxx in fxx_range:
            file_name = f"refs.t{cycle:02d}z.ffri.f{fxx:02d}.conus.grib2"
            grib_url = f"{base_url}{folder_path}/{file_name}"
            idx_url = f"{grib_url}.idx"
            local_file = refs_download_dir / f"refs_subset_f{fxx:02d}.grib2"

            download_success = False
            for attempt in range(3):
                if download_aws_subset(grib_url, idx_url, ":PPFFG:", local_file):
                    download_success = True
                    break
                time.sleep(2)

            if not download_success: continue

            for idx_file in refs_download_dir.glob('*.idx'):
                try: idx_file.unlink()
                except: pass

            try:
                hour_max = None
                for N in [1, 3, 6]:
                    start_hour = fxx - N
                    if start_hour < ero_start_fxx: continue

                    target_step = f"{start_hour}-{fxx}"
                    try:
                        ds_interval = xr.open_dataset(local_file, engine="cfgrib",
                                                      backend_kwargs={'filter_by_keys': {'stepRange': target_step}})
                        temp_data = ds_interval.ppffg.values
                        if hour_max is None: hour_max = temp_data
                        else: hour_max = np.maximum(hour_max, temp_data)

                        if lats is None:
                            lats = ds_interval.latitude.values
                            lons = ds_interval.longitude.values
                        ds_interval.close()
                    except: continue

                if hour_max is not None:
                    if max_ffg_grid is None: max_ffg_grid = hour_max
                    else: max_ffg_grid = np.maximum(max_ffg_grid, hour_max)
            except: pass

        refs_results[target_day] = {
            'max': max_ffg_grid, 'lats': lats, 'lons': lons,
            'cycle': cycle, 'status': status
        }
    except ValueError as e:
        print(f"  -> {e} Skipping Day {target_day}...")

# ==========================================
# 3. PLOTTING AND EXPORT
# ==========================================

if href_results and refs_results:
    print("\n--- GENERATING GRAPHICS ---")
    ero_colors = ['#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#A52A2A', '#FF00FF']
    cmap = ListedColormap(ero_colors)
    bounds = [5, 15, 25, 40, 55, 70, 100]
    norm = BoundaryNorm(bounds, cmap.N)
    cmap.set_over('#FF00FF')
    proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)

    for day in [1, 2]:
        fig, axes = plt.subplots(1, 2, figsize=(24, 10), subplot_kw={'projection': proj})

        href = href_results.get(day, {})
        refs = refs_results.get(day, {})

        for ax in axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, linewidth=1.0)
            ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
            ax.set_extent([-120, -70, 20, 50], crs=ccrs.PlateCarree())

        # --- PLOT HEAVILY SMOOTHED HREF ---
        h_max_raw, h_lats, h_lons = href.get('max'), href.get('lats'), href.get('lons')
        h_cycle, h_stat = href.get('cycle', 0), href.get('status', 'Unknown')
        h_max_smooth = apply_heavy_smoothing(h_max_raw)

        if h_max_smooth is not None and np.nanmax(h_max_smooth) >= 5:
            axes[0].contourf(h_lons, h_lats, h_max_smooth, transform=ccrs.PlateCarree(), levels=bounds, cmap=cmap, norm=norm, extend='max')
        else:
            axes[0].text(0.5, 0.5, "NO FLASH FLOOD THREAT\nOR DATA UNAVAILABLE",
                         transform=axes[0].transAxes, fontsize=25, color='red', alpha=0.3, fontweight='bold', ha='center', va='center')

        axes[0].set_title(f"HREF {h_cycle:02d}z Smoothed Max FFG Exceedance\nValid: Day {day} ERO Period ({h_stat})",
                          fontsize=16, loc='left', fontweight='bold')

        # --- PLOT HEAVILY SMOOTHED REFS ---
        r_max_raw, r_lats, r_lons = refs.get('max'), refs.get('lats'), refs.get('lons')
        r_cycle, r_stat = refs.get('cycle', 0), refs.get('status', 'Unknown')
        r_max_smooth = apply_heavy_smoothing(r_max_raw)

        if r_max_smooth is not None and np.nanmax(r_max_smooth) >= 5:
            axes[1].contourf(r_lons, r_lats, r_max_smooth, transform=ccrs.PlateCarree(), levels=bounds, cmap=cmap, norm=norm, extend='max')
        else:
            axes[1].text(0.5, 0.5, "NO FLASH FLOOD THREAT\nOR DATA UNAVAILABLE",
                         transform=axes[1].transAxes, fontsize=25, color='red', alpha=0.3, fontweight='bold', ha='center', va='center')

        axes[1].set_title(f"REFS {r_cycle:02d}z Smoothed Max FFG Exceedance\nValid: Day {day} ERO Period ({r_stat})",
                          fontsize=16, loc='left', fontweight='bold')

        # --- COLORBAR & FORMATTING ---
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=bounds)
        cbar.ax.set_xticklabels([f'{t}%' for t in bounds], fontsize=13)
        cbar.set_label(f'Day {day} - Maximum Probability of Exceeding FFG (1/3/6hr) [Smoothed]', fontsize=15, fontweight='bold')

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        
# Save output and clear memory
        archive_dir = Path("archive")
        archive_dir.mkdir(exist_ok=True)
        
        # 1. Save the Archive Copy
        archive_filename = f"archive/{date_str}_{h_cycle:02d}z_day{day}.png"
        plt.savefig(archive_filename, bbox_inches='tight', dpi=150)
        
        # 2. Save the Latest Copy
        latest_filename = f"day{day}_latest.png"
        plt.savefig(latest_filename, bbox_inches='tight', dpi=150)
        
        print(f"Saved {latest_filename} and archived as {archive_filename}")
        plt.close(fig) # Prevent memory leaks in Github Actions
else:
    print("Missing data: Processing failed for one or both models.")