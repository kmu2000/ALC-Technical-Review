# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xarray as xr
import rioxarray
import os
import pandas as pd
import geopandas as gpd
from rasterstats import point_query
from scipy.stats import pearsonr, spearmanr
from rasterio.enums import Resampling


def nctotif(ncdata, outdir):
    name = os.path.basename(ncdata)[:-3]
    target_crs = "EPSG:27700"  
    # Open dataset
    ds = xr.open_dataset(ncdata)

    # Pick variable
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    # Remove time dimension if it's size 1
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0)

    # Drop any multi-dimensional coordinate variables (e.g., lat/lon grids)
    for coord in list(da.coords):
        if da[coord].ndim > 1:
            da = da.drop_vars(coord)

    # Identify spatial dims
    if "projection_x_coordinate" in da.dims and "projection_y_coordinate" in da.dims:
        da = da.rio.set_spatial_dims(x_dim="projection_x_coordinate", y_dim="projection_y_coordinate")
    elif "longitude" in da.dims and "latitude" in da.dims:
        da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    else:
        raise ValueError(f"Cannot find spatial dimensions in {da.dims}")

    # Write CRS of source data
    da = da.rio.write_crs("EPSG:27700")  # HadUK-Grid native CRS; change if different

    # Handle NoData
    nodata_val = da.attrs.get("_FillValue", da.attrs.get("missing_value", -9999))
    da = da.rio.write_nodata(nodata_val)

    # Reproject to target CRS
    da_reprojected = da.rio.reproject(target_crs)

    # Output path for 1 km raster
    outpath = os.path.join(outdir, name + ".tif")

    # Save as GeoTIFF
    da_reprojected.rio.to_raster(outpath)
    
    # Reproject to 5 km using averaging
    da_5km = da_reprojected.rio.reproject(
        target_crs,
        resolution=5000,              
        resampling=Resampling.average
    )

    # Output path for 5 km raster
    outpath5 = os.path.join(outdir, name + "_5km.tif")
    da_5km.rio.to_raster(outpath5)
    
    return outpath, outpath5
    
      

def extractfromraster(inraster, gdf, name):
        
    # extract nearest rainfall values from the rainfall raster
    gdf[name] = point_query(gdf, inraster, interpolate='nearest')

    # Save updated shapefile or CSV
    gdf.to_file(os.path.join('../Data',"ALC_climate_compare.shp")) 
    
    return gdf
    

def testnormal(df, name1, name2):
    '''use this test to check whether the distribution is normal'''
    from scipy.stats import kstest

    # Normalize your data (optional but often done for numerical stability)
    data = df[name1]
    z = (data - data.mean()) / data.std()
    
    # Test if the data follows a standard normal distribution
    stat1, p1 = kstest(z, 'norm')
    # print(f"KS test statistic for AAR: {stat1:.4f}, p-value: {p1:.4f}")
    
    
    data = df[name2]
    z = (data - data.mean()) / data.std()
    
    num_nans = df[name2].isna().sum()
    if num_nans>0:
        data = df[name2].dropna()
    
    z = (data - data.mean()) / data.std()
    
    # Test if the data follows a standard normal distribution
    stat2, p2 = kstest(z, 'norm')
    # print(f"KS test statistic for AA_n: {stat2:.4f}, p-value: {p2:.4f}")
    
       
    return p1, p2

def testdistdiff(df, var1, var2):
    '''use this test to check whether one distribution tends to have larger values than the other.
    This is a nonparametric version of the parametric t test. We use this test when the distributions 
    we compare are not normal.'''
    from scipy.stats import mannwhitneyu
    
    df = df[[var1, var2]].dropna()
            
    stat, p = mannwhitneyu(df[var1], df[var2], alternative='two-sided')
    # print(f"Mann-Whitney U statistic: {stat}, p-value: {p}")
    
    return p

def comparestat(df, var1, var2):
    AAR_median = df[var1].median()
    AARhuk_median = df[var2].median()

    IQR_AAR = df[var1].quantile(0.75) - df[var1].quantile(0.25)
    IQR_AAR_huk = df[var2].quantile(0.75) - df[var2].quantile(0.25)
    
    return AAR_median, AARhuk_median, IQR_AAR, IQR_AAR_huk

# Format p-values in normal decimal style (max 4 decimals)
def format_p(p):
    if p < 0.0001:
        return "<0.0001"
    else:
        return f"{p:.4f}"

def plotscatter(df, var1, var2):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    df = df.dropna()
    x = df[var1]
    y = df[var2]   
    
    bias = np.mean(x-y)
    rmse = np.sqrt(np.mean((x-y)**2))
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)
    
    # Print to console
    print(f"Pearson correlation: {pearson_corr:.3f}, p-value: {pearson_p:.3e}")
    print(f"Spearman correlation: {spearman_corr:.3f}, p-value: {spearman_p:.3e}")
    
    # Plot style
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(7, 5))
    
    # Scatter with regression line
    ax = sns.regplot(x=x, y=y, scatter_kws={'s':60, 'alpha':0.7}, line_kws={'color':'red'})
    
    # Ensure equal axis limits for 1:1 line
    lims = [
        min(x.min(), y.min()),
        max(x.max(), y.max())
    ]
    ax.plot(lims, lims, '--', color='black', linewidth=1)  # 1:1 line
    ax.set_xlim(lims)
    ax.set_ylim(lims)
        
    # Annotate correlation on plot
    textstr = '\n'.join((
        f"Pearson r = {pearson_corr:.2f} p={format_p(pearson_p)}",
        f"Spearman ρ = {spearman_corr:.2f} p={format_p(pearson_p)}"
    ))
    
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))
    
    # Labels and title
    plt.xlabel(var1, fontsize=14)
    plt.ylabel(var2, fontsize=14)
    plt.title(f"Scatter Plot: {var1} vs {var2}", fontsize=16, weight='bold')
    
    plt.tight_layout()
    # Save as image
    plt.savefig("../Results/" + var1 + "_comparison.png", dpi=300)
    plt.close()  
    return bias, rmse
    

def plot_distributions(data, var1, var2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    sns.histplot(data[var1], label='Met Office', color='blue', kde=True, stat='density', alpha=0.5)
    sns.histplot(data[var2], label='HadUK', color='orange', kde=True, stat='density', alpha=0.5)
    plt.xlabel(var1 + " (mm)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()    
    
    # Save as image
    plt.savefig("../results/" + var1 + "distributions.png", dpi=300)
    plt.close()    

def plot_kde_difference(data, var1, var2):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import gaussian_kde

    # KDE estimation
    kde1 = gaussian_kde(data[var1].dropna())
    kde2 = gaussian_kde(data[var2].dropna())

    # Define range
    x = np.linspace(min(data[var1].min(), data[var2].min()),
                    max(data[var1].max(), data[var2].max()), 1000)

    # Evaluate KDEs
    y1 = kde1(x)
    y2 = kde2(x)
    diff = y1 - y2

    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, label='Met Office', color='blue')
    plt.plot(x, y2, label='HadUK', color='orange')
    plt.fill_between(x, diff, 0, where=diff > 0, color='blue', alpha=0.3, label='Met > HadUK')
    plt.fill_between(x, diff, 0, where=diff < 0, color='orange', alpha=0.3, label='HadUK > Met')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel(var1 + " (mm)")
    plt.ylabel("Density")
    plt.title("KDE and Density Difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../Results/" + var1 + "_kde_difference.png", dpi=300)
    plt.close()

def plot_cdfs(data, var1, var2):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 5))
    for label, series, color in [('Met Office', data[var1], 'blue'), ('HadUK', data[var2], 'orange')]:
        sorted_data = np.sort(series.dropna())
        yvals = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        plt.plot(sorted_data, yvals, label=label, color=color)
    
    plt.title("Cumulative Distribution Comparison")
    plt.xlabel(var1 + " (mm)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../Results/" + var1 + "_cdf_comparison.png", dpi=300)
    plt.close()


# open the average rainfall and temperature data
path = '../Data/ClimateData/HadUK_daily/temperature_1991_2020/AT'
AT0_HUK = os.path.join(path, 'AT0.nc')
ATS_HUK = os.path.join(path, 'ATS.nc')
# convert the netcdf to tif
AT0, AT05 = nctotif(AT0_HUK, path)
ATS, ATS5 = nctotif(ATS_HUK, path)

# read the shape file with MetOffice climate data 5 km
if os.path.exists('../Data/ALC_climate_compare.shp'):
    MOdata = '../Data/ALC_climate_compare.shp'
else:
    MOdata = '../Data/ALC_climate_loc_mo.shp'
gdf = gpd.read_file(MOdata)

# extract the AT0 and ATS values nearest the grid points 
gdf = extractfromraster(AT05, gdf, name='AT0_HUK5')
gdf = extractfromraster(ATS5, gdf, name='ATS_HUK5')
'''
# comapre the distributions of rainfall data
data = gpd.read_file('../data/ALC_climate_compare.shp')
p10, p20 = testnormal(data, name1='AT0', name2='AT0_HUK')
p1S, p2S = testnormal(data, name1='ATS', name2='ATS_HUK')

# check whether one distribution tends to have larger values than the other
p0 = testdistdiff(data, var1='AT0', var2='AT0_HUK')
pS = testdistdiff(data, var1='ATS', var2='ATS_HUK')

# compare the medians and IQRs
med10, med20, IQR10, IQR20 = comparestat(data, var1='AT0', var2='AT0_HUK')
med1S, med2S, IQR1S, IQR2S = comparestat(data, var1='ATS', var2='ATS_HUK')

os.makedirs("../Results", exist_ok=True)
outfile = '../Results/AT_comparison.txt'

# scatter plot and correlation
b0, r0 = plotscatter(data, var1='AT0', var2='AT0_HUK')
bS, rS = plotscatter(data, var1='ATS', var2='ATS_HUK')


# compare the distributions
plot_distributions(data, var1='AT0', var2='AT0_HUK')
plot_distributions(data, var1='ATS', var2='ATS_HUK')
plot_kde_difference(data, var1='AT0', var2='AT0_HUK')
plot_kde_difference(data, var1='ATS', var2='ATS_HUK')
plot_cdfs(data, var1='AT0', var2='AT0_HUK')
plot_cdfs(data, var1='ATS', var2='ATS_HUK')

with open(outfile, 'w') as f:
    f.write(f'mean bias between ATO Met Office and HadUK: {b0}\n')
    f.write(f'RMSE between AT0 Met Office and HadUK: {r0}\n\n')
    f.write(f'mean bias between ATS Met Office and HadUK: {bS}\n')
    f.write(f'RMSE between ATS Met Office and HadUK: {rS}\n\n')
    if p10<0.05:
        f.write('The Met Office AT0 does not have a normal distribution\n')
    else:
        f.write('The Met Office AT0 has a normal distribution\n')
        
    if p20<0.05:
        f.write('The Met Office ATS does not have a normal distribution\n')
    else:
        f.write('The Met Office ATS has a normal distribution\n')
        
    if p1S<0.05:
        f.write('The AT0_HUK does not have a normal distribution\n')
    else:
        f.write('The AT0_HUK has a normal distribution\n')
        
    if p2S<0.05:
        f.write('The ATS_HUK rainfall does not have a normal distribution\n')
    else:
        f.write('The ATS_HUK rainfall has a normal distribution\n')
    
    if p0<0.05:
        f.write('The two distributions of AT0 for Met Office and HadUK (1961-1980) differ significantly\n')
    else:
        f.write('The two distributions of AT0 for Met Office and HadUK (1961-1980) do not differ significantly\n')
        
    if pS<0.05:
        f.write('The two distributions of ATS for Met Office and HadUK (1961-1980) differ significantly\n')
    else:
        f.write('The two distributions of ATS for Met Office and HadUK (1961-1980) do not differ significantly\n')
        
    
    f.write(f"Met Office ATO Median: {med10:.2f}\n")
    f.write(f"HadUK AT0 Median: {med20:.2f}\n")
    f.write(f"Met Office AT0 IQR: {IQR10:.2f}\n")
    f.write(f"HadUK AT0 IQR: {IQR20:.2f}\n")
    
    f.write(f"Met Office ATS Median: {med1S:.2f}\n")
    f.write(f"HadUK ATS Median: {med2S:.2f}\n")
    f.write(f"Met Office ATS IQR: {IQR1S:.2f}\n")
    f.write(f"HadUK ATS IQR: {IQR2S:.2f}\n")
    
'''