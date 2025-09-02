# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:29:16 2025

@author: kriti.mukherjee
"""

import os
import xarray as xr
import numpy as np
import calendar

# Paths to your tasmax and tasmin folders
path_tasmax = '../Data/HadUK_daily/temperature/tasmax/'
path_tasmin = '../Data/HadUK_daily/temperature/tasmin/'

# Output folder for results
output_path = '../Data/HadUK_daily/temperature/AT1/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

years = range(1991, 2021)
months = range(1, 13)

for y in years:
    for m in months:
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        
        # Filenames
        file_max = f"tasmax_hadukgrid_uk_1km_day_{start_date}-{end_date}.nc"
        file_min = f"tasmin_hadukgrid_uk_1km_day_{start_date}-{end_date}.nc"
        
        path_max = os.path.join(path_tasmax, file_max)
        path_min = os.path.join(path_tasmin, file_min)
        
        if os.path.isfile(path_max) and os.path.isfile(path_min):
            print(f"Processing {start_date}-{end_date}")
            
            # Open datasets (xarray will mask fill values automatically)
            ds_max = xr.open_dataset(path_max)
            ds_min = xr.open_dataset(path_min)
            
            tasmax = ds_max['tasmax']
            tasmin = ds_min['tasmin']
            
            # Mask out invalid pixels (only compute where both are valid)
            valid_mask = (~tasmax.isnull()) & (~tasmin.isnull())
            tasmean = ((tasmax + tasmin) / 2.0).where(valid_mask)
            
            # Keep only positive daily averages
            tasmean_positive = tasmean.where(tasmean > 0)
            
            # Sum over time ignoring NaNs
            monthly_sum = tasmean_positive.sum(dim='time', skipna=True)
            
            # Apply domain mask to set always-invalid cells to NaN
            domain_mask = valid_mask.any(dim='time')
            monthly_sum = monthly_sum.where(domain_mask)
            
            # Copy attributes and set coordinates explicitly
            monthly_sum.attrs = tasmax.attrs.copy()
            if 'coordinates' in tasmax.attrs:
                monthly_sum.attrs['coordinates'] = tasmax.attrs['coordinates']  # Preserve CF coordinates
            monthly_sum.attrs['units'] = 'degree_Celsius_days'
            monthly_sum.attrs['long_name'] = 'Monthly sum of positive daily mean temperatures'
            
            # Create dataset
            out_ds = monthly_sum.to_dataset(name='tasmean_positive_sum')
            out_ds.attrs.update(ds_max.attrs)  # Copy global attributes
            
            # Filter valid encoding parameters
            valid_encodings = {'shuffle', 'dtype', 'fletcher32', 'zlib', 'contiguous', 'complevel', 'chunksizes', 'least_significant_digit', '_FillValue'}
            encoding = {k: v for k, v in tasmax.encoding.items() if k in valid_encodings}
            
            # NEW: Adjust chunksizes for 2D output (remove time dimension)
            if 'chunksizes' in encoding:
                # Assume input chunksizes are (time, y, x); take only (y, x)
                input_chunks = encoding['chunksizes']
                if len(input_chunks) == 3:  # Typical for (time, projection_y_coordinate, projection_x_coordinate)
                    encoding['chunksizes'] = input_chunks[1:]  # Keep only y, x
                else:
                    del encoding['chunksizes']  # Remove if unexpected format
            
            # Assign encoding for the output variable
            out_encoding = {'tasmean_positive_sum': encoding}
            
            # Save result
            out_file = os.path.join(output_path, f"tasmean_positive_sum_{start_date}-{end_date}.nc")
            out_ds.to_netcdf(out_file, encoding=out_encoding)
            
            ds_max.close()
            ds_min.close()
        else:
            print(f"Missing files for {start_date}-{end_date}")

# Now compute AT0: sum Jan-Jun per year, then median over years
print("Computing AT0...")
at0_yearly_sums = []
years_list = list(years)
for y in years_list:
    yearly_sum = None
    for m in range(1, 7):
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        file = f"tasmean_positive_sum_{start_date}-{end_date}.nc"
        path = os.path.join(output_path, file)
        if os.path.isfile(path):
            ds = xr.open_dataset(path)
            data = ds['tasmean_positive_sum']
            if yearly_sum is None:
                yearly_sum = data
            else:
                yearly_sum += data
            ds.close()
        else:
            print(f"Missing monthly file for AT0: {file}")
            yearly_sum = None
            break
    if yearly_sum is not None:
        at0_yearly_sums.append(yearly_sum)

if at0_yearly_sums:
    at0_da = xr.concat(at0_yearly_sums, dim='year')
    at0_da.coords['year'] = years_list[:len(at0_yearly_sums)]  # Adjust if any years missing
    at0_median = at0_da.median(dim='year', skipna=True)
    
    # Create dataset, copy attributes from an example monthly file
    example_file = os.path.join(output_path, "tasmean_positive_sum_19910101-19910131.nc")  # Assuming this exists
    if os.path.isfile(example_file):
        example_ds = xr.open_dataset(example_file)
        out_ds_at0 = at0_median.to_dataset(name='AT0')
        out_ds_at0.attrs.update(example_ds.attrs)
        out_ds_at0['AT0'].attrs['units'] = 'degree_Celsius_days'
        out_ds_at0['AT0'].attrs['long_name'] = 'Median over years of sum of positive daily mean temperatures from Jan to Jun'
        
        # Encoding similar to monthly
        encoding = {k: v for k, v in example_ds['tasmean_positive_sum'].encoding.items() if k in valid_encodings}
        out_encoding = {'AT0': encoding}
        
        out_ds_at0.to_netcdf(os.path.join(output_path, 'AT0.nc'), encoding=out_encoding)
        example_ds.close()
    else:
        print("Example monthly file not found for AT0 attributes.")
else:
    print("No complete yearly sums for AT0.")

# Now compute ATS: sum Apr-Sep per year, then median over years
print("Computing ATS...")
ats_yearly_sums = []
for y in years_list:
    yearly_sum = None
    for m in range(4, 10):
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        file = f"tasmean_positive_sum_{start_date}-{end_date}.nc"
        path = os.path.join(output_path, file)
        if os.path.isfile(path):
            ds = xr.open_dataset(path)
            data = ds['tasmean_positive_sum']
            if yearly_sum is None:
                yearly_sum = data
            else:
                yearly_sum += data
            ds.close()
        else:
            print(f"Missing monthly file for ATS: {file}")
            yearly_sum = None
            break
    if yearly_sum is not None:
        ats_yearly_sums.append(yearly_sum)

if ats_yearly_sums:
    ats_da = xr.concat(ats_yearly_sums, dim='year')
    ats_da.coords['year'] = years_list[:len(ats_yearly_sums)]  # Adjust if any years missing
    ats_median = ats_da.median(dim='year', skipna=True)
    
    # Create dataset, copy attributes from an example monthly file
    if os.path.isfile(example_file):
        example_ds = xr.open_dataset(example_file)
        out_ds_ats = ats_median.to_dataset(name='ATS')
        out_ds_ats.attrs.update(example_ds.attrs)
        out_ds_ats['ATS'].attrs['units'] = 'degree_Celsius_days'
        out_ds_ats['ATS'].attrs['long_name'] = 'Median over years of sum of positive daily mean temperatures from Apr to Sep'
        
        # Encoding similar to monthly
        encoding = {k: v for k, v in example_ds['tasmean_positive_sum'].encoding.items() if k in valid_encodings}
        out_encoding = {'ATS': encoding}
        
        out_ds_ats.to_netcdf(os.path.join(output_path, 'ATS.nc'), encoding=out_encoding)
        example_ds.close()
    else:
        print("Example monthly file not found for ATS attributes.")
else:
    print("No complete yearly sums for ATS.")