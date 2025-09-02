# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:08:21 2025

@author: kriti.mukherjee
"""

import os
import calendar
import subprocess


pathweb = 'https://data.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.3.1.ceda/1km/rainfall/day/v20250415/'  # change for other variables
pathnc = '/mnt/c/Users/Kriti.Mukherjee/Cranfield University/ALC Technical Review-UGT-FEAS - Documents/General/Data/ClimateData/HadUK_daily/rainfall_1991_2021/'

if not os.path.exists(pathnc):
    os.makedirs(pathnc)

# generate a new access token each time you download the data. It is valid for 3 days.
access_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI4ZjhmaUpyaUtDY3hmaHhzdU5vazVEekdJdFZ4amhhTWNJa05ZX2U4MnhJIn0.eyJleHAiOjE3NTY4OTA0ODcsImlhdCI6MTc1NjYzMTI4NywianRpIjoiZjU0OWMzYmMtZTJiYS00NWMzLWI1NGYtNGE0ZTNmOGZmY2Y1IiwiaXNzIjoiaHR0cHM6Ly9hY2NvdW50cy5jZWRhLmFjLnVrL3JlYWxtcy9jZWRhIiwic3ViIjoiZmFiZTNhMWMtOGUzMi00Y2QxLWI1Y2YtMWNkYzE3NGMxNjI4IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoic2VydmljZXMtcG9ydGFsLWNlZGEtYWMtdWsiLCJzZXNzaW9uX3N0YXRlIjoiZmRhZDBkZmMtYWI1Ni00ZGQ4LWEzZDQtMmRmNWIzYTExNTBhIiwiYWNyIjoiMSIsInNjb3BlIjoiZW1haWwgb3BlbmlkIHByb2ZpbGUgZ3JvdXBfbWVtYmVyc2hpcCIsInNpZCI6ImZkYWQwZGZjLWFiNTYtNGRkOC1hM2Q0LTJkZjViM2ExMTUwYSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJvcGVuaWQiOiJodHRwczovL2NlZGEuYWMudWsvb3BlbmlkL0tyaXRpLk11a2hlcmplZTEiLCJuYW1lIjoiS3JpdGkgTXVraGVyamVlIiwicHJlZmVycmVkX3VzZXJuYW1lIjoia211a2hlcmplZTAwMSIsImdpdmVuX25hbWUiOiJLcml0aSIsImZhbWlseV9uYW1lIjoiTXVraGVyamVlIiwiZW1haWwiOiJrcml0aS5tdWtoZXJqZWVAY3JhbmZpZWxkLmFjLnVrIn0.cMnSujR1sLeBSYDYyVqGbGDeXMRAPvChJbfNU_t5IEpuQWMoCa7cgoklzJu6rDSDpgc9Empt21tFk9oJ668vNcpsQuU7kQVuQTZyRhvvcuxRR00hIkXK4KwB-UsU8XaFLb73xtfGjNUsyP6fZelbEt3d9F3OiDK1TwUEAbjyBWsAQUCN-9kHAreMwPA3RqywFy5eUhc_EBGYG_KCdK3ad-LIdLgSbRL5vjb_xF2RL6tDXJsoyOG3FMEGwSSlrJaY9R5qGDysyYUzuZgDo7akVlCFzy4ZmcgjBiFnD4YYssIlQu-N-NJhrsb4JuJafPGuroDdoEDRGVMtj5jOnqPrYQ"

# Your access token
# access_token = os.environ.get("TOKEN") 
year = range(1991,2021)
month = range(1,13)

 
data_prefix = 'rainfall_hadukgrid_uk_1km_day_' # change the prefix for other variables 

for y in year:
    for m in month:
        last_day = calendar.monthrange(y, m)[1]
        start_date = f"{y}{m:02d}01"
        end_date = f"{y}{m:02d}{last_day:02d}"
        dataname = f"{data_prefix}{start_date}-{end_date}.nc"
        local_file = os.path.join(pathnc, dataname)

        if not os.path.isfile(local_file):
            print("Downloading:", dataname)
            cmd = [
                "wget",
                "--header", f"Authorization: Bearer {access_token}",
                f"{pathweb}{dataname}",
                "-O", local_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error downloading:", dataname)
                print(result.stderr)
        else:
            print("Already exists:", local_file)