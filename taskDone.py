import streamlit as st
from streamlit_folium import st_folium

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import folium


acceleration_data = pd.read_csv('Linear Acceleration.csv')
location_data = pd.read_csv('Location.csv')

from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

#T is the length of the signal
#last element minus first element means total time duration
t = acceleration_data['Time (s)'].iloc[-1] - acceleration_data['Time (s)'].iloc[0]
st.write(f'Time duration of the signal: {t} seconds')

fs = len(acceleration_data['Time (s)']) / t  # Sampling frequency
nyq = 0.5 * fs  # Nyquist Frequency
order = 3
cutoff_low = 2.5  # Desired cutoff frequency of the filter, Hz

filtered_z_low = butter_lowpass_filter(acceleration_data['Linear Acceleration z (m/s^2)'], cutoff_low, fs, nyq, order)


st.line_chart(acceleration_data['Linear Acceleration z (m/s^2)'])
st.line_chart(filtered_z_low)

stepcount = 0

#calculate step count from filtered signal by finding local maxima
for i in range(1, len(filtered_z_low) - 1):
    if((filtered_z_low[i] > 0 and filtered_z_low[i] > filtered_z_low[i - 1] and filtered_z_low[i] > filtered_z_low[i + 1])):
        stepcount += 1

st.write(f"Total steps counted: {stepcount}.")


#calculate step count from fourier transformed signal

N = len(acceleration_data['Linear Acceleration z (m/s^2)'])

yf = fft(acceleration_data['Linear Acceleration z (m/s^2)'])
xf = fftfreq(N, 1 / fs)

st.line_chart({'Frequency (Hz)': xf[:N // 2],
'Amplitude': 2.0 / N * np.abs(yf[:N // 2])})



#find the peak frequency in the fourier transformed signal
amplitudes = 2.0 / N * np.abs(yf[:N // 2])
peak_freq_index = np.argmax(amplitudes)
peak_frequency = xf[peak_freq_index]
print(f"Peak frequency in the Fourier Transform: {peak_frequency} Hz")

#calculate step count from peak frequency
calculated_steps_fourier = peak_frequency * t
print(f"Calculated steps: {calculated_steps_fourier}.")



#use haversine formula for the distance calculation between two gps coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


distance = 0.0

for i in range(1, len(location_data['Latitude (°)'])):
    lat1 = location_data['Latitude (°)'][i - 1]
    lon1 = location_data['Longitude (°)'][i - 1]
    lat2 = location_data['Latitude (°)'][i]
    lon2 = location_data['Longitude (°)'][i]
    distance += haversine(lat1, lon1, lat2, lon2)

st.write(f'Total distance traveled (in meters): {distance * 1000} meters')


#let's calculate average speed
average_speed = (distance * 1000) / t  # in meters per second
st.write(f'Average speed: {average_speed} m/s')




map = folium.Map(location=[location_data['Latitude (°)'].mean(), location_data['Longitude (°)'].mean()], zoom_start=14)

for i in range(len(location_data['Latitude (°)'])):
    folium.CircleMarker(
        location=[location_data['Latitude (°)'][i], location_data['Longitude (°)'][i]],
        radius=2,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(map)

folium.PolyLine(location_data[['Latitude (°)', 'Longitude (°)']], color='red').add_to(map)
map.save('location_map.html')

st_map = st_folium(map, width=700, height=500)