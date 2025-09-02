# ALC Technical Review Data Processing

This repository contains scripts for downloading, processing, and comparing climate data (temperature and rainfall) with ALC baseline datasets.

---

## Steps

### 1. Download Daily Temperature Data
Use the script **`download_HadUKDaily.py`** to download `tasmin` and `tasmax`.

- **1961–1980**: To match with ALC baseline data.  
- **1991–2020**: To get the latest baseline data.  

👉 To run this script, you must obtain an **access token** linked to your CEDA credentials.  
See [CEDA Help: Downloading multiple files with wget](https://help.ceda.ac.uk/article/5191-downloading-multiple-files-with-wget) for details.

---

### 2. Convert to AT0 and ATS
Use the script **`calcAT.py`** to convert downloaded NetCDF files into AT0 and ATS variables.

---

### 3. Compare with Met Office AT0 and ATS
Run **`AT_compare_MO.py`** to compare calculated values with Met Office AT0 and ATS datasets.

---

### 4. Download Daily Rainfall
Use the script **`download_HadUKDaily.py`** to download daily rainfall.

- **1941–1970**: To match with ALC baseline data.  
- **1991–2020**: To get the latest baseline data.  

---

### 5. Convert to AAR and ASR
Use **`calcAR.py`** to convert rainfall data into AAR and ASR variables.

---

### 3. Compare with Met Office AAR and ASR
Run **`AR_compare_MO.py`** to compare calculated values with Met Office AAR and ASR datasets.

---

## Notes
- All scripts should be run in an environment with access to NetCDF libraries (e.g., `xarray`, `netCDF4`).  
- Ensure your CEDA account is active and you have the correct download permissions before running the data download scripts.  
