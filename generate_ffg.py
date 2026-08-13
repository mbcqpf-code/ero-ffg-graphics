import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import xarray as xr
import requests
import time
import os
import urllib.request
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter, convolve
from scipy.spatial import cKDTree
from herbie import Herbie
import cfgrib

warnings.filterwarnings('ignore')

# ==========================================
# 1. HELPER FUNCTIONS & MASTER CONFIG
# ==========================================
GRID_RES_KM = 3.0
NEIGHBORHOOD_KM = 40.0
radius_pts = int(NEIGHBORHOOD_KM / GRID_RES_KM)
y_grid, x_grid = np.ogrid[-radius_pts : radius_pts + 1, -radius_pts : radius_pts + 1]
circular_footprint = x_grid**2 + y_grid**2 <= radius_pts**2

def get_fxx_range(cycle, target_day):
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
    now = datetime.now(timezone.utc)
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
                return date_str, cycle, fxx_range, status, dt
        except: pass
    raise ValueError(f"Could not find fully uploaded HREF runs for Day {target_day}.")

def get_rrfs_fxx_range(cycle, target_day):
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
    now = datetime.now(timezone.utc)
    current_cycle_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    
    # REFS Shifted to NOMADS Para Servers
    base_url = "https://nomads.ncep.noaa.gov"
    headers = {"User-Agent": "Mozilla/5.0"}

    for i in range(6):
        dt = current_cycle_time - timedelta(hours=6 * i)
        cycle = dt.hour
        date_str = dt.strftime("%Y%m%d")
        fxx_range, status = get_rrfs_fxx_range(cycle, target_day)
        if fxx_range is None: continue
        last_fxx = fxx_range[-1]

        folder_path = f"/pub/data/nccf/com/refs/para/refs.{date_str}/{cycle:02d}"
        file_name = f"refs.t{cycle:02d}z.ffri.f{last_fxx:02d}.conus.grib2"
        idx_url = f"{base_url}{folder_path}/{file_name}.idx"

        try:
            if requests.head(idx_url, headers=headers, timeout=5).status_code == 200:
                print(f"✅ Locked in fully uploaded REFS run: Date={date_str}, Cycle={cycle:02d}z")
                return date_str, cycle, fxx_range, folder_path, base_url, status
        except: pass
    raise ValueError(f"Could not find fully uploaded REFS runs.")

def download_idx_subset(grib_url, idx_url, search_str, local_file):
    # User-Agent added to prevent NOMADS from blocking headless servers
    headers = {"User-Agent": "Mozilla/5.0"}
    idx_resp = requests.get(idx_url, headers=headers, timeout=10)
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
    
    req_headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={min_byte}-{max_byte}"}
    grib_resp = requests.get(grib_url, headers=req_headers, timeout=30)
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
# 2. DATA PROCESSING LOOP
# ==========================================
href_results = {}
refs_results = {}
href_pmm_results = {}
href_qpf_results = {}

Path("href_downloads").mkdir(exist_ok=True)
Path("refs_downloads").mkdir(exist_ok=True)
Path("ffg_data").mkdir(exist_ok=True)

for target_day in [1, 2]:
    # ----------------------------------------------------
    # A. PPFFG HREF EXCEEDANCE PROBABILITIES
    # ----------------------------------------------------
    print(f"\n--- PROCESSING HREF DAY {target_day} ---")
    try:
        today_date, cycle, fxx_range, status, cycle_dt = get_latest_href_run(target_day)
        max_ffg_grid, lats, lons = None, None, None
        ero_start_fxx = fxx_range[0]

        for fxx in fxx_range:
            file_name = f"href.t{cycle:02d}z.conus.ffri.f{fxx:02d}.grib2"
            dir_path = f"/href.{today_date}/ensprod"
            url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"
            params = {"dir": dir_path, "file": file_name, "var_PPFFG": "on", "all_lev": "on"}
            local_file = Path("href_downloads") / f"{file_name}_subset"

            download_success = False
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                    if response.status_code == 200 and len(response.content) > 1000:
                        with open(local_file, 'wb') as f: f.write(response.content)
                        download_success = True
                        break
                    else: time.sleep(5) 
                except: time.sleep(5)
            if not download_success: continue

            for idx_file in Path("href_downloads").glob('*.idx'):
                try: idx_file.unlink()
                except: pass

            try:
                hour_max = None
                for N in [1, 3, 6]:
                    start_hour = fxx - N
                    if start_hour < ero_start_fxx: continue
                    target_step = f"{start_hour}-{fxx}"
                    try:
                        ds_interval = xr.open_dataset(local_file, engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepRange': target_step}})
                        temp_data = ds_interval.ppffg.values
                        hour_max = temp_data if hour_max is None else np.maximum(hour_max, temp_data)
                        if lats is None:
                            lats = ds_interval.latitude.values
                            lons = ds_interval.longitude.values
                        ds_interval.close()
                    except: continue

                if hour_max is not None:
                    max_ffg_grid = hour_max if max_ffg_grid is None else np.maximum(max_ffg_grid, hour_max)
            except: pass
            time.sleep(1)

        href_results[target_day] = {
            'max': max_ffg_grid, 'lats': lats, 'lons': lons,
            'cycle': cycle, 'status': status, 'date': today_date
        }

        # ----------------------------------------------------
        # B. HREF PMM QPF VS IEM FFG (RATIO & COVERAGE)
        # ----------------------------------------------------
        print(f"\n--- FETCHING IEM FFG & HREF PMM (DAY {target_day}) ---")
        iem_date_path = cycle_dt.strftime('%Y/%m/%d')
        iem_file_time = cycle_dt.strftime('%Y%m%d%H')
        iem_ffg_url = f"https://mesonet.agron.iastate.edu/archive/data/{iem_date_path}/model/ffg/5kmffg_{iem_file_time}.grib2"
        local_ffg_path = f"ffg_data/5kmffg_{iem_file_time}.grib2"
        
        if not os.path.exists(local_ffg_path):
            urllib.request.urlretrieve(iem_ffg_url, local_ffg_path)
            
        ds_1h = xr.open_dataset(local_ffg_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepRange': '0-1'}})
        ds_3h = xr.open_dataset(local_ffg_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepRange': '0-3'}})
        ds_6h = xr.open_dataset(local_ffg_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepRange': '0-6'}})
        var_name = list(ds_1h.data_vars)[0]
        ffg_1h_da = ds_1h[var_name] / 25.4
        ffg_3h_da = ds_3h[var_name] / 25.4
        ffg_6h_da = ds_6h[var_name] / 25.4

        run_date_herbie = cycle_dt.strftime("%Y-%m-%d %H:00")
        href_pmm_1hr_list = []
        for fxx in fxx_range:
            if fxx == fxx_range[0]: continue 
            try:
                H_pmm = Herbie(run_date_herbie, model="href", product="pmmn", domain="conus", fxx=fxx)
                H_pmm.download()
                ds_pmm = xr.open_dataset(H_pmm.get_localFilePath(), engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "tp", "stepRange": f"{fxx-1}-{fxx}"}})
                href_pmm_1hr_list.append(ds_pmm["tp"] / 25.4)
            except: pass

        if len(href_pmm_1hr_list) > 0:
            qpf_1h_da = xr.concat(href_pmm_1hr_list, dim="time").fillna(0)
            lat_grid = qpf_1h_da.latitude.values
            lon_grid = qpf_1h_da.longitude.values
            
            ffg_points = np.column_stack((ffg_1h_da.latitude.values.ravel(), ffg_1h_da.longitude.values.ravel()))
            tree = cKDTree(ffg_points)
            href_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
            _, indices = tree.query(href_points)
            ffg_1h_aligned = ffg_1h_da.values.ravel()[indices].reshape(lat_grid.shape)
            ffg_3h_aligned = ffg_3h_da.values.ravel()[indices].reshape(lat_grid.shape)
            ffg_6h_aligned = ffg_6h_da.values.ravel()[indices].reshape(lat_grid.shape)

            ratio_1h_max = np.zeros(lat_grid.shape)
            ratio_3h_max = np.zeros(lat_grid.shape)
            ratio_6h_max = np.zeros(lat_grid.shape)

            for t in range(qpf_1h_da.sizes['time']):
                q1 = qpf_1h_da.isel(time=t).values
                safe_ffg_1h = np.where(ffg_1h_aligned > 0, ffg_1h_aligned, np.nan)
                with np.errstate(invalid='ignore', divide='ignore'): ratio_1h_max = np.fmax(ratio_1h_max, q1 / safe_ffg_1h)

                if t >= 2:
                    q3 = qpf_1h_da.isel(time=slice(t-2, t+1)).sum(dim='time').values
                    safe_ffg_3h = np.where(ffg_3h_aligned > 0, ffg_3h_aligned, np.nan)
                    with np.errstate(invalid='ignore', divide='ignore'): ratio_3h_max = np.fmax(ratio_3h_max, q3 / safe_ffg_3h)

                if t >= 5:
                    q6 = qpf_1h_da.isel(time=slice(t-5, t+1)).sum(dim='time').values
                    safe_ffg_6h = np.where(ffg_6h_aligned > 0, ffg_6h_aligned, np.nan)
                    with np.errstate(invalid='ignore', divide='ignore'): ratio_6h_max = np.fmax(ratio_6h_max, q6 / safe_ffg_6h)

            max_ratio_overall = np.fmax(ratio_1h_max, np.fmax(ratio_3h_max, ratio_6h_max))
            
            max_ratio_40km = maximum_filter(max_ratio_overall, footprint=circular_footprint)
            ratio_masked = np.where(max_ratio_40km >= 0.75, max_ratio_40km, np.nan)
            
            binary_exceedance = np.where(max_ratio_overall >= 1.0, 1.0, 0.0)
            coverage_grid = convolve(binary_exceedance, circular_footprint, mode='constant', cval=0.0)
            coverage_fraction = (coverage_grid / np.sum(circular_footprint)) * 100.0
            coverage_masked = np.where(coverage_fraction >= 1.0, coverage_fraction, np.nan)
            
            href_pmm_results[target_day] = {
                'ratio': ratio_masked, 'coverage': coverage_masked,
                'lats': lat_grid, 'lons': lon_grid,
                'cycle': cycle, 'date': today_date, 'status': status
            }

        # ----------------------------------------------------
        # C. NATIVE 24-HR QPF (1" EAS + 3" NEP)
        # ----------------------------------------------------
        print(f"\n--- FETCHING 24-HR QPF (DAY {target_day}) ---")
        skip_qpf = False
        if target_day == 1 and cycle == 18: skip_qpf = True
        if target_day == 2 and cycle in [0, 6]: skip_qpf = True

        if not skip_qpf:
            try:
                if target_day == 1:
                    if cycle == 0: fxx_end, acc_regex = 36, r"12-36"
                    elif cycle == 6: fxx_end, acc_regex = 30, r"6-30"
                    elif cycle == 12: fxx_end, acc_regex = 24, r"(0-24|0-1.+)"
                else:
                    if cycle == 12: fxx_end, acc_regex = 48, r"(24-48|1-2.+)"
                    elif cycle == 18: fxx_end, acc_regex = 42, r"18-42"
                
                H_eas = Herbie(run_date_herbie, model="href", product="eas", domain="conus", fxx=fxx_end)
                H_prob = Herbie(run_date_herbie, model="href", product="prob", domain="conus", fxx=fxx_end)
                
                search_eas_1in = rf"APCP:surface:{acc_regex}.*:prob >25.4"
                search_prob_3in = rf"APCP:surface:{acc_regex}.*:prob >76.2"

                ds_eas_1in = H_eas.xarray(search_eas_1in)
                ds_prob_3in = H_prob.xarray(search_prob_3in)
                
                eas_1in_vals = ds_eas_1in[list(ds_eas_1in.data_vars)[0]].values
                nep_3in_vals = ds_prob_3in[list(ds_prob_3in.data_vars)[0]].values
                
                href_qpf_results[target_day] = {
                    'eas': np.where(eas_1in_vals >= 15, eas_1in_vals, np.nan),
                    'nep': nep_3in_vals,
                    'lats': ds_eas_1in.latitude.values,
                    'lons': ds_eas_1in.longitude.values,
                    'cycle': cycle, 'date': today_date
                }
            except Exception as e:
                print(f"  -> Failed 24-hr QPF for Day {target_day}: {e}")

    except ValueError as e:
        print(f"  -> {e} Skipping Day {target_day}...")

    # ----------------------------------------------------
    # D. PPFFG REFS EXCEEDANCE PROBABILITIES
    # ----------------------------------------------------
    print(f"\n--- PROCESSING REFS DAY {target_day} ---")
    try:
        date_str, cycle, fxx_range, folder_path, base_url, status = get_latest_rrfs_run(target_day)
        max_ffg_grid, lats, lons = None, None, None
        ero_start_fxx = fxx_range[0]

        for fxx in fxx_range:
            file_name = f"refs.t{cycle:02d}z.ffri.f{fxx:02d}.conus.grib2"
            grib_url = f"{base_url}{folder_path}/{file_name}"
            idx_url = f"{grib_url}.idx"
            local_file = Path("refs_downloads") / f"refs_subset_f{fxx:02d}.grib2"

            download_success = False
            for attempt in range(3):
                if download_idx_subset(grib_url, idx_url, ":PPFFG:", local_file):
                    download_success = True
                    break
                time.sleep(2)
            if not download_success: continue

            for idx_file in Path("refs_downloads").glob('*.idx'):
                try: idx_file.unlink()
                except: pass

            try:
                hour_max = None
                for N in [1, 3, 6]:
                    start_hour = fxx - N
                    if start_hour < ero_start_fxx: continue
                    target_step = f"{start_hour}-{fxx}"
                    try:
                        ds_interval = xr.open_dataset(local_file, engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepRange': target_step}})
                        temp_data = ds_interval.ppffg.values
                        hour_max = temp_data if hour_max is None else np.maximum(hour_max, temp_data)
                        if lats is None:
                            lats = ds_interval.latitude.values
                            lons = ds_interval.longitude.values
                        ds_interval.close()
                    except: continue

                if hour_max is not None:
                    max_ffg_grid = hour_max if max_ffg_grid is None else np.maximum(max_ffg_grid, hour_max)
            except: pass

        refs_results[target_day] = {
            'max': max_ffg_grid, 'lats': lats, 'lons': lons,
            'cycle': cycle, 'status': status, 'date': date_str
        }
    except ValueError as e:
        print(f"  -> {e} Skipping Day {target_day}...")

# ==========================================
# 3. PLOTTING AND EXPORT
# ==========================================
Path("archive").mkdir(exist_ok=True)
proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)

if href_results and refs_results:
    print("\n--- GENERATING GRAPHICS ---")
    ero_colors = ['#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#A52A2A', '#FF00FF']
    cmap_prob = ListedColormap(ero_colors)
    bounds_prob = [5, 15, 25, 40, 55, 70, 100]
    norm_prob = BoundaryNorm(bounds_prob, cmap_prob.N)
    cmap_prob.set_over('#FF00FF')

    for day in [1, 2]:
        # --- GRAPHIC 1: HREF vs REFS EXCEEDANCE PROBABILITY ---
        fig, axes = plt.subplots(1, 2, figsize=(24, 10), subplot_kw={'projection': proj})
        href = href_results.get(day, {})
        refs = refs_results.get(day, {})

        for ax in axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, linewidth=1.0)
            ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
            ax.set_extent([-120, -70, 20, 50], crs=ccrs.PlateCarree())

        h_max_raw, h_lats, h_lons = href.get('max'), href.get('lats'), href.get('lons')
        h_cycle, h_stat, h_date = href.get('cycle', 0), href.get('status', 'Unknown'), href.get('date', 'UNKNOWN_DATE')
        h_max_smooth = apply_heavy_smoothing(h_max_raw)

        if h_max_smooth is not None and np.nanmax(h_max_smooth) >= 5:
            axes[0].contourf(h_lons, h_lats, h_max_smooth, transform=ccrs.PlateCarree(), levels=bounds_prob, cmap=cmap_prob, norm=norm_prob, extend='max')
        
        axes[0].set_title(f"HREF {h_date} {h_cycle:02d}z Smoothed Max FFG Exceedance\nValid: Day {day} ERO Period ({h_stat})", fontsize=16, loc='left', fontweight='bold')

        r_max_raw, r_lats, r_lons = refs.get('max'), refs.get('lats'), refs.get('lons')
        r_cycle, r_stat, r_date = refs.get('cycle', 0), refs.get('status', 'Unknown'), refs.get('date', 'UNKNOWN_DATE')
        r_max_smooth = apply_heavy_smoothing(r_max_raw)

        if r_max_smooth is not None and np.nanmax(r_max_smooth) >= 5:
            axes[1].contourf(r_lons, r_lats, r_max_smooth, transform=ccrs.PlateCarree(), levels=bounds_prob, cmap=cmap_prob, norm=norm_prob, extend='max')

        axes[1].set_title(f"REFS {r_date} {r_cycle:02d}z Smoothed Max FFG Exceedance\nValid: Day {day} ERO Period ({r_stat})", fontsize=16, loc='left', fontweight='bold')

        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        sm = ScalarMappable(cmap=cmap_prob, norm=norm_prob)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=bounds_prob)
        cbar.ax.set_xticklabels([f'{t}%' for t in bounds_prob], fontsize=13)
        cbar.set_label(f'Day {day} - Maximum Probability of Exceeding FFG (1/3/6hr) [Smoothed]', fontsize=15, fontweight='bold')
        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        
        plt.savefig(f"archive/{h_date}_{h_cycle:02d}z_day{day}.png", bbox_inches='tight', dpi=150)
        plt.savefig(f"day{day}_latest.png", bbox_inches='tight', dpi=150)
        plt.close(fig) 
        
        # --- GRAPHIC 2: PMM MAGNITUDE RATIO VS FRACTIONAL COVERAGE ---
        if day in href_pmm_results:
            pmm = href_pmm_results[day]
            fig_pmm, axes_pmm = plt.subplots(1, 2, figsize=(24, 10), subplot_kw={'projection': proj})
            
            for ax in axes_pmm:
                ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
                ax.add_feature(cfeature.BORDERS, linewidth=1.0)
                ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
                ax.set_extent([-120, -70, 20, 50], crs=ccrs.PlateCarree())

            ratio_levels = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            ratio_colors = ['#ffff00', '#ffa500', '#ff0000', '#8b0000', '#ff00ff', '#800080', '#0000ff', '#00ffff']
            cf_ratio = axes_pmm[0].contourf(pmm['lons'], pmm['lats'], pmm['ratio'], levels=ratio_levels, colors=ratio_colors, transform=ccrs.PlateCarree(), alpha=0.9, extend='max')
            axes_pmm[0].set_title(f"HREF pmmn {pmm['date']} {pmm['cycle']:02d}z: Max FFG Exceedance Ratio\n40-km Circular Neighborhood", fontsize=16, fontweight='bold', loc='left')
            cbar_ratio = plt.colorbar(cf_ratio, ax=axes_pmm[0], orientation='horizontal', pad=0.03, shrink=0.8, aspect=40)
            cbar_ratio.set_label('Exceedance Ratio (QPF / FFG)', fontsize=12, fontweight='bold')

            coverage_levels = [1, 5, 10, 25, 50, 75, 100]
            coverage_colors = ['#e0f7fa', '#c8e6c9', '#fff59d', '#ffb74d', '#f44336', '#9c27b0'] 
            cf_cov = axes_pmm[1].contourf(pmm['lons'], pmm['lats'], pmm['coverage'], levels=coverage_levels, colors=coverage_colors, transform=ccrs.PlateCarree(), alpha=0.9, extend='max')
            axes_pmm[1].set_title(f"HREF pmmn {pmm['date']} {pmm['cycle']:02d}z: FFG Exceedance Coverage\n40-km Neighborhood Fraction", fontsize=16, fontweight='bold', loc='left')
            cbar_cov = plt.colorbar(cf_cov, ax=axes_pmm[1], orientation='horizontal', pad=0.03, shrink=0.8, aspect=40)
            cbar_cov.set_label('Areal Coverage Percentage (%)', fontsize=12, fontweight='bold')
            
            plt.subplots_adjust(bottom=0.05, wspace=0.05)
            plt.savefig(f"archive/{pmm['date']}_{pmm['cycle']:02d}z_day{day}_pmm.png", bbox_inches='tight', dpi=150)
            plt.savefig(f"day{day}_pmm_latest.png", bbox_inches='tight', dpi=150)
            plt.close(fig_pmm)

        # --- GRAPHIC 3: DUAL-SIGNAL 24-HR QPF ---
        if day in href_qpf_results:
            qpf = href_qpf_results[day]
            fig_qpf, ax_qpf = plt.subplots(1, 1, figsize=(14, 10), subplot_kw={'projection': proj})
            ax_qpf.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax_qpf.add_feature(cfeature.BORDERS, linewidth=1.0)
            ax_qpf.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
            ax_qpf.set_extent([-120, -70, 20, 50], crs=ccrs.PlateCarree())

            eas_levels = [15, 30, 50, 70, 90, 100]
            eas_colors = ["#c8e6c9", "#fff59d", "#ffb74d", "#ff8a65", "#d32f2f"]
            cf_eas = ax_qpf.contourf(qpf['lons'], qpf['lats'], qpf['eas'], levels=eas_levels, colors=eas_colors, transform=ccrs.PlateCarree(), alpha=0.85)

            nep_levels = [15, 40, 70]
            nep_colors = ["#6a1b9a", "#e91e63", "#000000"]
            cs_nep = ax_qpf.contour(qpf['lons'], qpf['lats'], qpf['nep'], levels=nep_levels, colors=nep_colors, linewidths=1.2, transform=ccrs.PlateCarree())

            cbar_eas = plt.colorbar(cf_eas, ax=ax_qpf, orientation="horizontal", pad=0.03, shrink=0.7, aspect=40)
            cbar_eas.set_label('Native 1" EAS Probability (%) [Filled Contours]', fontsize=12, fontweight="bold")

            custom_lines = [Line2D([0], [0], color=c, lw=1.5) for c in nep_colors]
            legend = ax_qpf.legend(custom_lines, [f"{lvl}%" for lvl in nep_levels], title='3" NEP (Lines)', loc="lower right", framealpha=0.95, fontsize=11, title_fontsize=12)
            legend.get_frame().set_edgecolor("black")

            ax_qpf.set_title(f"HREF {qpf['date']} {qpf['cycle']:02d}z: 1\" EAS Coverage + 3\" NEP Magnitude\nValid: Day {day} 24-hr QPF", fontsize=16, fontweight="bold", loc="left")
            
            plt.savefig(f"archive/{qpf['date']}_{qpf['cycle']:02d}z_day{day}_qpf.png", bbox_inches='tight', dpi=150)
            plt.savefig(f"day{day}_qpf_latest.png", bbox_inches='tight', dpi=150)
            plt.close(fig_qpf)
        else:
            if h_cycle != 0: 
                fig_blank, ax_blank = plt.subplots(figsize=(14, 10))
                ax_blank.text(0.5, 0.5, f"NO 24-HR QPF AVAILABLE\nDay {day} is a partial period for the {h_cycle:02d}z cycle.", fontsize=25, color='gray', alpha=0.5, fontweight='bold', ha='center', va='center')
                ax_blank.axis('off')
                plt.savefig(f"archive/{h_date}_{h_cycle:02d}z_day{day}_qpf.png", bbox_inches='tight', dpi=150)
                plt.savefig(f"day{day}_qpf_latest.png", bbox_inches='tight', dpi=150)
                plt.close(fig_blank)

else:
    print("Missing data: Processing failed for one or both models.")
