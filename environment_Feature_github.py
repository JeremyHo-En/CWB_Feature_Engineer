#%%
import pandas as pd
import glob
import numpy as np

#%%
#READ
weather_data = pd.read_csv("./CWB_weather_199912to202312.csv")
weather_data['localTime_TW'] = pd.to_datetime(weather_data['localTime_TW'])
air_data = pd.read_csv('./air_monitoring.csv', encoding='utf-8')
air_data['datetime'] = pd.to_datetime(air_data['datetime'])
all_data = pd.merge(weather_data, air_data, left_on='localTime_TW' , right_on='datetime', how='left')

#%%
#CLEAN
all_data = all_data.drop(columns=['PS01_467490.1','PP01_467490.1','TX01_467490.1','RH01_467490.1','WD01_467490.1','WD02_467490.1'])
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].where(all_data[numeric_cols] >= -1, np.nan)
all_data[numeric_cols].mask(all_data[numeric_cols] < 0, np.nan, inplace=True)
all_data[numeric_cols] = all_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
#%%
#Add Time Tag
all_data['date'] = all_data['localTime_TW'].dt.date
all_data['month'] = all_data['localTime_TW'].dt.strftime('%Y-%m')
all_data['day'] = all_data['localTime_TW'].dt.date
all_data['week'] = all_data['localTime_TW'].dt.strftime('%Y-W%W')
all_data['week_add_day'] = all_data['localTime_TW'].dt.strftime('%G-W%V-%u')
#%%
def calculate_wind_chill(temp, wind):
    return 13.12 + 0.6215 * temp - 11.37 * (wind ** 0.16) + 0.3965 * temp * (wind ** 0.16)

def calculate_heat_index(temp, humidity):
    return -8.784695 + 1.61139411 * temp + 2.338549 * humidity - 0.14611605 * temp * humidity

def calculate_mode(x):
    x = x.dropna()
    mode_series = x.mode()
    return mode_series.iloc[0] if not mode_series.empty else np.nan

def WD_find(trace,dataty):

    trace_rounded = trace.copy()
    trace_rounded[stationlist_WD] = trace_rounded[stationlist_WD].apply(lambda col: col.map(lambda x: round(x / 10) * 10 if pd.notnull(x) and x > 0 else np.nan))

    wind_mode_per_station = (trace_rounded.groupby([dataty])[stationlist_WD].agg(calculate_mode).reset_index())

    wind_mode_final = wind_mode_per_station.copy()
    wind_mode_final['final_mode'] = wind_mode_per_station[stationlist_WD].apply(lambda row: calculate_mode(row), axis=1)

    wind_mode_final = wind_mode_final[[dataty, 'final_mode']]

    return wind_mode_final

def Weather_Feature(trace,airlist,dataty):

    #make list
    stationlist_Temp = [col for col in trace.columns if 'TX01' in col]
    stationlist_Pressure = [col for col in trace.columns if 'PS01' in col]
    stationlist_Rain = [col for col in trace.columns if 'PP01' in col]
    stationlist_Wind = [col for col in trace.columns if 'WD01' in col]
    stationlist_RH = [col for col in trace.columns if 'RH01' in col]
    stationlist_WD = [col for col in trace.columns if 'WD02' in col]

    #Time Feature
    new = (trace.groupby([dataty])[stationlist_Temp].mean().reset_index())[dataty].to_frame(name=dataty)

    #Temp Feature
    new['Temp_avg'] = (trace.groupby([dataty])[stationlist_Temp].mean().reset_index())[stationlist_Temp].mean(axis=1)
    new['Temp_max'] = (trace.groupby([dataty])[stationlist_Temp].max().reset_index())[stationlist_Temp].mean(axis=1)
    new['Temp_min'] = (trace.groupby([dataty])[stationlist_Temp].min().reset_index())[stationlist_Temp].mean(axis=1)
    new['Temp_diff'] = new['Temp_max']-new['Temp_min']

    day_max = trace.groupby(['date'])[stationlist_Temp].max().reset_index()
    day_max[dataty] = (trace.groupby(['date'])[dataty].first().reset_index())[dataty]
    day_min = trace.groupby(['date'])[stationlist_Temp].min().reset_index()
    day_min[dataty] = (trace.groupby(['date'])[dataty].first().reset_index())[dataty]
    day_diff = day_max[stationlist_Temp]-day_min[stationlist_Temp]
    day_diff[dataty] = (trace.groupby(['date'])[dataty].first().reset_index())[dataty]
    new['Temp_max_avg'] = (day_max.groupby([dataty])[stationlist_Temp].mean().reset_index())[stationlist_Temp].mean(axis=1)
    new['Temp_min_avg'] = (day_min.groupby([dataty])[stationlist_Temp].mean().reset_index())[stationlist_Temp].mean(axis=1)
    new['Temp_diff_avg'] = (day_diff.groupby([dataty])[stationlist_Temp].mean().reset_index())[stationlist_Temp].mean(axis=1)

    day_max['day_avg'] = day_max[stationlist_Temp].mean(axis=1)
    new = new.merge(day_max[day_max['day_avg'] >= 34].groupby([dataty]).size().reset_index(name='hot_34'), on=dataty, how='left')
    day_min['day_avg'] = day_min[stationlist_Temp].mean(axis=1)
    new = new.merge(day_min[day_min['day_avg'] <= 10].groupby([dataty]).size().reset_index(name='cold_10'), on=dataty, how='left')
    day_diff['day_avg'] = day_diff[stationlist_Temp].mean(axis=1)
    new = new.merge(day_diff[day_diff['day_avg'] >= 10].groupby([dataty]).size().reset_index(name='Tdiff_10'), on=dataty, how='left')
    new = new.fillna(0)

    #Pressure Feature
    new['PS_avg'] = (trace.groupby([dataty])[stationlist_Pressure].mean().reset_index())[stationlist_Pressure].mean(axis=1)
    new['PS_max'] = (trace.groupby([dataty])[stationlist_Pressure].max().reset_index())[stationlist_Pressure].mean(axis=1)
    new['PS_min'] = (trace.groupby([dataty])[stationlist_Pressure].min().reset_index())[stationlist_Pressure].mean(axis=1)
    new['PS_diff'] = new['PS_max']-new['PS_min']
    
    day_max = trace.groupby(['date'])[stationlist_Pressure].max().reset_index()
    day_max[dataty] = (trace.groupby(['date'])[dataty].first().reset_index())[dataty]
    day_min = trace.groupby(['date'])[stationlist_Pressure].min().reset_index()
    day_min[dataty] = (trace.groupby(['date'])[dataty].first().reset_index())[dataty]
    day_diff = day_max[stationlist_Pressure]-day_min[stationlist_Pressure]
    day_diff[dataty] = (trace.groupby(['date'])[dataty].first().reset_index())[dataty]
    new['PS_max_avg'] = (day_max.groupby([dataty])[stationlist_Pressure].mean().reset_index())[stationlist_Pressure].mean(axis=1)
    new['PS_min_avg'] = (day_min.groupby([dataty])[stationlist_Pressure].mean().reset_index())[stationlist_Pressure].mean(axis=1)
    new['PS_diff_avg'] = (day_diff.groupby([dataty])[stationlist_Pressure].mean().reset_index())[stationlist_Pressure].mean(axis=1)

    #Rain Feature
    new['PP_sum'] = (trace.groupby([dataty])[stationlist_Rain].sum().reset_index())[stationlist_Rain].mean(axis=1)

    #Air Feature
    all_data_exclude_date = all_data.loc[:, all_data.columns != 'date']
    station_namelist = ['Songshan', 'Qianzhe', 'ChungMing']

    for col in airlist:
        stationlist_col = [f"{col}_{station_name}" for station_name in station_namelist]
        valid_columns = [c for c in stationlist_col if c in trace.columns]
        
        if not valid_columns:
            print(f"Warning: No valid columns found for {col}")
            continue
        
        new[f'{col}_mean'] = trace.groupby([dataty])[valid_columns].mean().reset_index()[valid_columns].mean(axis=1)
        new[f'{col}_max'] = trace.groupby([dataty])[valid_columns].max().reset_index()[valid_columns].mean(axis=1)
        new[f'{col}_min'] = trace.groupby([dataty])[valid_columns].min().reset_index()[valid_columns].mean(axis=1)
        new[f'{col}_diff'] = new[f'{col}_max'] - new[f'{col}_min']

        day_max = trace.groupby(['date'])[valid_columns].max().reset_index()
        day_max[dataty] = trace.groupby(['date'])[dataty].first().values
        
        day_min = trace.groupby(['date'])[valid_columns].min().reset_index()
        day_min[dataty] = trace.groupby(['date'])[dataty].first().values
        
        day_diff = day_max[valid_columns] - day_min[valid_columns]
        day_diff[dataty] = trace.groupby(['date'])[dataty].first().values
        
        new[f'{col}_max_avg'] = day_max.groupby([dataty])[valid_columns].mean().reset_index()[valid_columns].mean(axis=1)
        new[f'{col}_min_avg'] = day_min.groupby([dataty])[valid_columns].mean().reset_index()[valid_columns].mean(axis=1)
        new[f'{col}_diff_avg'] = day_diff.groupby([dataty])[valid_columns].mean().reset_index()[valid_columns].mean(axis=1)

    
    #Wind Feature
    new['Wind_avg'] = (trace.groupby([dataty])[stationlist_Wind].mean().reset_index())[stationlist_Wind].mean(axis=1)
    new['Wind_max'] = (trace.groupby([dataty])[stationlist_Wind].max().reset_index())[stationlist_Wind].mean(axis=1)

    #WD Feature
    new['WD_avg'] = (trace.groupby([dataty])[stationlist_WD].mean().reset_index())[stationlist_WD].mean(axis=1)
    new['WD_mode'] = WD_find(trace,dataty)['final_mode']

    #FLT Feature
    FLT_Temp = trace[['month']].copy()
    FLT_Temp['date'] = trace[['date']]

    for j, TPsta in enumerate(stationlist_Temp):
        WDsta = stationlist_Wind[j]
        RHsta = stationlist_RH[j]

        temp_c = trace[TPsta]
        wind_kmh = trace[WDsta]
        humidity = trace[RHsta]

        is_cold = (temp_c < 10) & (wind_kmh > 4.8)
        is_hot = (temp_c > 27) & (humidity > 40)

        FLT_Temp[f'FLT_{TPsta[5:]}'] = np.where(is_cold, calculate_wind_chill(temp_c, wind_kmh),np.where(is_hot, calculate_heat_index(temp_c, humidity), temp_c))
    
    stationlist_FLT = [col for col in FLT_Temp.columns if 'FLT_' in col]

    new['FLT_avg'] = (FLT_Temp.groupby([dataty])[stationlist_FLT].mean().reset_index())[stationlist_FLT].mean(axis=1)
    new['FLT_max'] = (FLT_Temp.groupby([dataty])[stationlist_FLT].max().reset_index())[stationlist_FLT].mean(axis=1)
    new['FLT_min'] = (FLT_Temp.groupby([dataty])[stationlist_FLT].min().reset_index())[stationlist_FLT].mean(axis=1)
    new['FLT_diff'] = new['FLT_max'] - new['FLT_min']
    day_max = FLT_Temp.groupby(['date'])[stationlist_FLT].max().reset_index()
    day_max[dataty] = FLT_Temp.groupby(['date'])[dataty].first().values
    day_min = FLT_Temp.groupby(['date'])[stationlist_FLT].min().reset_index()
    day_min[dataty] = FLT_Temp.groupby(['date'])[dataty].first().values
    day_diff = day_max[stationlist_FLT] - day_min[stationlist_FLT]
    day_diff[dataty] = FLT_Temp.groupby(['date'])[dataty].first().values
    new['FLT_max_avg'] = (day_max.groupby([dataty])[stationlist_FLT].mean().reset_index())[stationlist_FLT].mean(axis=1)
    new['FLT_min_avg'] = (day_min.groupby([dataty])[stationlist_FLT].mean().reset_index())[stationlist_FLT].mean(axis=1)
    new['FLT_diff_avg'] = (day_diff.groupby([dataty])[stationlist_FLT].mean().reset_index())[stationlist_FLT].mean(axis=1)



    if dataty == 'week':
        new['week_add_day'] = (trace.groupby([dataty])['week_add_day'].first().reset_index())['week_add_day']
        new['week_add_day'] =  pd.to_datetime(new['week_add_day'],format='%G-W%V-%u')
    else:
        new[dataty] =  pd.to_datetime(new[dataty])
    return new
# %%
airlist = ['CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'SO2']
Feature_array = Weather_Feature(all_data,airlist,'month')
# %%
new_columns = Feature_array.select_dtypes(include=[np.number]).columns
diff_df = Feature_array[[col for col in new_columns]].diff()
diff_df.columns = [f'{col}_Change' for col in diff_df.columns]
Feature_array = pd.concat([Feature_array, diff_df], axis=1)
# %%
Feature_array.to_csv('./Feature_array.csv', index=False, encoding='utf-8')
# %%
