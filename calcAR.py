# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 16:08:28 2025

@author: kriti.mukherjee
"""

import os
import xarray as xr
import numpy as np
import calendar

# Paths to your tasmax and tasmin folders
path_rain = '../Data/ClimateData/HadUK_daily/rainfall_1991_2020/'

# Output folder for results
output_path = '../Data/ClimateData/HadUK_daily/rainfall_1991_2020/AR/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create dataset, copy attributes from an example monthly file
example_file = os.path.join(output_path, "rainfall_19910101-19910131.nc") 

years = range(1991, 2021)
months = range(1, 13)

for y in years:
    for m in months:
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        
        # Filenames
        file_rain = f"rainfall_hadukgrid_uk_1km_day_{start_date}-{end_date}.nc"
        
        
        file_path_rain = os.path.join(path_rain, file_rain)
        
        
        if os.path.isfile(file_path_rain):
            print(f"Processing {start_date}-{end_date}")
            
            # Open datasets (xarray will mask fill values automatically)
            ds_rain = xr.open_dataset(file_path_rain)           
            
            rainfall = ds_rain['rainfall']
            
            
            # Mask out invalid pixels (only compute where both are valid)
            valid_mask = (~rainfall.isnull()) 
            
            
            # Sum over time ignoring NaNs
            monthly_sum = rainfall.sum(dim='time', skipna=True)
            
            # Apply domain mask to set always-invalid cells to NaN
            domain_mask = valid_mask.any(dim='time')
            monthly_sum = monthly_sum.where(domain_mask)
            
            # Copy attributes and set coordinates explicitly
            monthly_sum.attrs = rainfall.attrs.copy()
            if 'coordinates' in rainfall.attrs:
                monthly_sum.attrs['coordinates'] = rainfall.attrs['coordinates']  # Preserve CF coordinates
            monthly_sum.attrs['units'] = 'mm'
            monthly_sum.attrs['long_name'] = 'Monthly sum of rainfall'
            
            # Create dataset
            out_ds = monthly_sum.to_dataset(name='monthly_total_rainfall')
            out_ds.attrs.update(ds_rain.attrs)  # Copy global attributes
            
            # Filter valid encoding parameters
            valid_encodings = {'shuffle', 'dtype', 'fletcher32', 'zlib', 'contiguous', 'complevel', 'chunksizes', 'least_significant_digit', '_FillValue'}
            encoding = {k: v for k, v in rainfall.encoding.items() if k in valid_encodings}
            
            # NEW: Adjust chunksizes for 2D output (remove time dimension)
            if 'chunksizes' in encoding:
                # Assume input chunksizes are (time, y, x); take only (y, x)
                input_chunks = encoding['chunksizes']
                if len(input_chunks) == 3:  # Typical for (time, projection_y_coordinate, projection_x_coordinate)
                    encoding['chunksizes'] = input_chunks[1:]  # Keep only y, x
                else:
                    del encoding['chunksizes']  # Remove if unexpected format
            
            # Assign encoding for the output variable
            out_encoding = {'monthly_total_rainfall': encoding}
            
            # Save result
            out_file = os.path.join(output_path, f"rainfall_{start_date}-{end_date}.nc")
            out_ds.to_netcdf(out_file, encoding=out_encoding)
            
            ds_rain.close()
            
        else:
            print(f"Missing files for {start_date}-{end_date}")

# Now compute AT0: sum Jan-Jun per year, then median over years
print("Computing AAR...")
aar_yearly_sums = []
years_list = list(years)
for y in years_list:
    yearly_sum = None
    for m in range(1, 13):
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        file = f"rainfall_{start_date}-{end_date}.nc"
        path = os.path.join(output_path, file)
        if os.path.isfile(path):
            ds = xr.open_dataset(path)
            data = ds['monthly_total_rainfall']
            if yearly_sum is None:
                yearly_sum = data
            else:
                yearly_sum += data
            ds.close()
        else:
            print(f"Missing monthly file for rainfall: {file}")
            yearly_sum = None
            break
    if yearly_sum is not None:
        aar_yearly_sums.append(yearly_sum)

if aar_yearly_sums:
    aar_da = xr.concat(aar_yearly_sums, dim='year')
    aar_da.coords['year'] = years_list[:len(aar_yearly_sums)]  # Adjust if any years missing
    aar_mean = aar_da.mean(dim='year', skipna=True)
    
     # Assuming this exists
    if os.path.isfile(example_file):
        example_ds = xr.open_dataset(example_file)
        out_ds_aar = aar_mean.to_dataset(name='AAR')
        out_ds_aar.attrs.update(example_ds.attrs)
        out_ds_aar['AAR'].attrs['units'] = 'mm'
        out_ds_aar['AAR'].attrs['long_name'] = 'Average over years of annual average rainfall'
        
        # Encoding similar to monthly
        encoding = {k: v for k, v in example_ds['monthly_total_rainfall'].encoding.items() if k in valid_encodings}
        out_encoding = {'AAR': encoding}
        
        out_ds_aar.to_netcdf(os.path.join(output_path, 'AAR.nc'), encoding=out_encoding)
        example_ds.close()
    else:
        print("Example monthly file not found for AAR attributes.")
else:
    print("No complete yearly sums for AAR.")

# Now compute ATS: sum Apr-Sep per year, then median over years
print("Computing ASR...")
asr_yearly_sums = []
for y in years_list:
    yearly_sum = None
    for m in range(4, 10):
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        file = f"rainfall_{start_date}-{end_date}.nc"
        path = os.path.join(output_path, file)
        if os.path.isfile(path):
            ds = xr.open_dataset(path)
            data = ds['monthly_total_rainfall']
            if yearly_sum is None:
                yearly_sum = data
            else:
                yearly_sum += data
            ds.close()
        else:
            print(f"Missing monthly file for ASR: {file}")
            yearly_sum = None
            break
    if yearly_sum is not None:
        asr_yearly_sums.append(yearly_sum)

if asr_yearly_sums:
    asr_da = xr.concat(asr_yearly_sums, dim='year')
    asr_da.coords['year'] = years_list[:len(asr_yearly_sums)]  # Adjust if any years missing
    asr_mean = asr_da.mean(dim='year', skipna=True)
    
    # Create dataset, copy attributes from an example monthly file
    if os.path.isfile(example_file):
        example_ds = xr.open_dataset(example_file)
        out_ds_asr = asr_mean.to_dataset(name='ASR')
        out_ds_asr.attrs.update(example_ds.attrs)
        out_ds_asr['ASR'].attrs['units'] = 'mm'
        out_ds_asr['ASR'].attrs['long_name'] = 'Average over years of rainfall from Apr to Sep'
        
        # Encoding similar to monthly
        encoding = {k: v for k, v in example_ds['monthly_total_rainfall'].encoding.items() if k in valid_encodings}
        out_encoding = {'ASR': encoding}
        
        out_ds_asr.to_netcdf(os.path.join(output_path, 'ASR.nc'), encoding=out_encoding)
        example_ds.close()
    else:
        print("Example monthly file not found for ASR attributes.")
else:
    print("No complete yearly sums for ASR.")