# download control and ensemble GloFAS forecasts (version 4)

import cdsapi
import os

c = cdsapi.Client()

output_directory = "/Volumes/TOSHIBA EXT/flood aa/forecast data"

for year in range(2003,2024):
    for month in ['10']:
        for day in ['02','05','09','12','16','19','23','26','30']: # days will change depending on month selected
            for product_type in ['control_reforecast', 'ensemble_perturbed_reforecasts']:
                file_name = f'GloFAS_{year}-{month}-{day}_{product_type}.grib'
                file_path = os.path.join(output_directory, file_name)
                c.retrieve(
                    'cems-glofas-reforecast',
                    {
                        'system_version': 'version_4_0',
                        'hydrological_model': 'lisflood',
                        'product_type': product_type,
                        'variable': 'river_discharge_in_the_last_24_hours',
                        'hyear': str(year),
                        'hmonth': month,
                        'hday': day,
                        'leadtime_hour': [
                            '24', '48', '72',
                            '96', '120', '144',
                            '168', '192', '216',
                            '240', '264', '288',
                            '312', '336', '360',
                            '384', '408', '432',
                            '456', '480', '504',
                            '528', '552', '576',
                            '600', '624', '648',
                            '672', '696', '720',
                            '744', '768', '792',
                            '816', '840', '864',
                            '888', '912', '936',
                            '960', '984', '1008',
                            '1032', '1056', '1080',
                            '1104',
                        ],
                        'format': 'grib',
                        'area': [
                            0, 25, -28,
                            42,
                            ],
                        },
                    file_path)