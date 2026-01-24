"""
Simple IMERG Data Downloader
Downloads GPM IMERG precipitation data for a specified region and time period.

Usage: Set environment variables or edit defaults below, then run the script.
       Can be called from SLURM script with exported variables.
"""

import os
import subprocess
import datetime
from datetime import timedelta
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly

# =============================================================================
# CONFIGURATION - Read from environment variables or use defaults
# =============================================================================

# Region bounds (longitude/latitude) - Burkina Faso defaults
XMIN = float(os.environ.get('XMIN', '-5.5'))
XMAX = float(os.environ.get('XMAX', '2.5'))
YMIN = float(os.environ.get('YMIN', '9.0'))
YMAX = float(os.environ.get('YMAX', '17.0'))

# Output grid size
OUTPUT_HEIGHT = int(os.environ.get('OUTPUT_HEIGHT', '64'))
OUTPUT_WIDTH = int(os.environ.get('OUTPUT_WIDTH', '64'))

# Time period - parse from environment or use defaults
def parse_date(date_str, default):
    """Parse date string in format YYYY-MM-DD-HH-MM"""
    if date_str:
        parts = date_str.split('-')
        return datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]), 
                                  int(parts[3]), int(parts[4]), 0)
    return default
                        # datetime.datetime(2011, 4, 1, 0, 0, 0))

START_DATE = parse_date(os.environ.get('START_DATE'), 
                        datetime.datetime(2014, 6, 1, 0, 0, 0))
END_DATE = parse_date(os.environ.get('END_DATE'), 
                      datetime.datetime(2022, 12, 31, 0, 0, 0))

# Output folder
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', '/home1/ppatel2025/ldcast/imerg_data/')

# NASA PPS credentials
EMAIL = os.environ.get('EMAIL', 'aaravamudan2014@my.fit.edu')

# Server path
SERVER_PATH = os.environ.get('SERVER_PATH', 
                             'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/')

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_file(filename, server, email):
    """Download file from NASA PPS server using curl."""
    url = server + '/' + filename
    local_name = os.path.basename(filename)
    cmd = f'curl -sO -u {email}:{email} {url}'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    # Check if downloaded file is actually HTML (error page) instead of GeoTIFF
    if os.path.isfile(local_name):
        with open(local_name, 'rb') as f:
            header = f.read(100)
            if b'<!DOCTYPE' in header or b'<html' in header.lower():
                os.remove(local_name)
                return False
    return True

def read_and_warp(grid_file, xmin, ymin, xmax, ymax, req_height, req_width):
    """Read a grid file and warp it to the specified domain and size."""
    raw_grid = gdal.Open(grid_file, GA_ReadOnly)

    if raw_grid is None:
        raise RuntimeError(f"Failed to open {grid_file} - file may be corrupted or not a valid raster")

    # Adjust grid to global extent
    pre_ds = gdal.Translate('temp.tif', raw_grid, options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")
    
    gt = pre_ds.GetGeoTransform()
    no_data = 29999
    
    # Calculate pixel size for target dimensions
    pixel_size_x = (xmax - xmin) / req_width
    pixel_size_y = (ymax - ymin) / req_height
    
    # Warp to target resolution and extent
    ds = gdal.Warp('', pre_ds, srcNodata=no_data, srcSRS='EPSG:4326', dstSRS='EPSG:4326',
                   dstNodata='-9999', format='VRT', xRes=pixel_size_x, yRes=-pixel_size_y,
                   outputBounds=(xmin, ymin, xmax, ymax))
    
    warped_grid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize
    
    return warped_grid, new_nx, new_ny, new_gt, new_proj

def write_grid(grid_out_name, data_out, nx, ny, gt, proj):
    """Write processed data to GeoTIFF."""
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(grid_out_name, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    data_out.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(data_out, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds = None

def download_imerg(output_folder, start_date, end_date, xmin, ymin, xmax, ymax,
                   req_height, req_width, server, email):
    """Download and process IMERG data for the specified region and time period."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
    file_suffix = '.V07B.30min.tif'
    delta_time = timedelta(minutes=30)
    
    current_date = start_date
    final_date = end_date + timedelta(minutes=30)
    
    while current_date < final_date:
        initial_time_stmp = current_date.strftime('%Y%m%d-S%H%M%S')
        final_time = current_date + timedelta(minutes=29)
        final_time_stmp = final_time.strftime('E%H%M59')
        final_time_gridout = current_date + timedelta(minutes=30)
        folder = current_date.strftime('%Y/%m/')
        
        total_minutes = current_date.hour * 60 + current_date.minute
        date_stamp = f"{initial_time_stmp}-{final_time_stmp}.{total_minutes:04}"
        
        filename = folder + file_prefix + date_stamp + file_suffix
        local_filename = file_prefix + date_stamp + file_suffix
        grid_out_name = output_folder + 'imerg.' + final_time_gridout.strftime('%Y%m%d%H%M') + '.tif'
        
        print(f'Downloading {final_time_gridout.strftime("%Y-%m-%d %H:%M")}')
        
        if not os.path.isfile(grid_out_name):
            get_file(filename, server, email)
            
            if os.path.isfile(local_filename):
                try:
                    # Process and warp to target region/size
                    new_grid, nx, ny, gt, proj = read_and_warp(
                        local_filename, xmin, ymin, xmax, ymax, req_height, req_width
                    )
                    # Scale precipitation values
                    new_grid = new_grid * 0.1

                    # Write output
                    write_grid(grid_out_name, new_grid, nx, ny, gt, proj)
                except RuntimeError as e:
                    print(f'  Warning: {e}')
                finally:
                    # Cleanup temp files
                    if os.path.isfile(local_filename):
                        os.remove(local_filename)
                    if os.path.isfile('temp.tif'):
                        os.remove('temp.tif')
            else:
                print(f'  Warning: Failed to download {local_filename}')
        else:
            print(f'  File already exists, skipping')
        
        current_date += delta_time
    
    print(f'\nDownload complete! Files saved to: {output_folder}')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("IMERG Data Downloader")
    print("="*60)
    print(f"Region: [{XMIN}, {XMAX}] x [{YMIN}, {YMAX}]")
    print(f"Grid size: {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output: {OUTPUT_FOLDER}")
    print("="*60)
    
    download_imerg(
        output_folder=OUTPUT_FOLDER,
        start_date=START_DATE,
        end_date=END_DATE,
        xmin=XMIN,
        ymin=YMIN,
        xmax=XMAX,
        ymax=YMAX,
        req_height=OUTPUT_HEIGHT,
        req_width=OUTPUT_WIDTH,
        server=SERVER_PATH,
        email=EMAIL
    )