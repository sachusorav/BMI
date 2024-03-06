import serial
import numpy as np
from scipy import signal
import joblib
import re
import time
import scipy
from scipy import signal
import pandas as pd
from collections import Counter
import pyautogui
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def calculate_psd_features(segment, sampling_rate):
    f, psd_values = scipy.signal.welch(segment, fs=sampling_rate, nperseg=len(segment))

    alpha_indices = np.where((f >= 8) & (f <= 13))
    beta_indices = np.where((f >= 14) & (f <= 30))
    theta_indices = np.where((f >= 4) & (f <= 7))
    delta_indices = np.where((f >= 0.5) & (f <= 3))

    energy_alpha = np.sum(psd_values[alpha_indices])
    energy_beta = np.sum(psd_values[beta_indices])
    energy_theta = np.sum(psd_values[theta_indices])
    energy_delta = np.sum(psd_values[delta_indices])

    alpha_beta_ratio = energy_alpha / energy_beta

    # Return features as a flat dictionary
    return {
        'E_alpha': energy_alpha,
        'E_beta': energy_beta,
        'E_theta': energy_theta,
        'E_delta': energy_delta,
        'alpha_beta_ratio': alpha_beta_ratio
    }

def calculate_additional_features(segment, sampling_rate):
    f, psd = scipy.signal.welch(segment, fs=sampling_rate, nperseg=len(segment))

    # Peak frequency
    peak_frequency = f[np.argmax(psd)]

    # Spectral centroid
    spectral_centroid = np.sum(f * psd) / np.sum(psd)

    # Spectral slope
    log_f = np.log(f[1:])
    log_psd = np.log(psd[1:])
    spectral_slope = np.polyfit(log_f, log_psd, 1)[0]

    return {
        'peak_frequency': peak_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_slope': spectral_slope
    }
scaler = joblib.load('scaler3.joblib')
features = []
columns = ['E_alpha', 'E_beta', 'E_theta', 'E_delta', 'alpha_beta_ratio','peak_frequency','spectral_centroid','spectral_slope']
# Load the trained model
clf = joblib.load('svm_model3.joblib')

# Serial connection to Arduino
ser = serial.Serial('COM6', baudrate=115200)  # Replace 'COMX' with your Arduino port

sampling_rate = 512
notch_freq = 50.0
lowcut, highcut = 0.5, 30.0

#  notch filter
nyquist = 0.5 * sampling_rate
notch_freq_normalized = notch_freq / nyquist
b_notch, a_notch = signal.iirnotch(notch_freq_normalized, Q=0.05, fs=sampling_rate)

#  bandpass filter
lowcut_normalized = lowcut / nyquist
highcut_normalized = highcut / nyquist
b_bandpass, a_bandpass = signal.butter(4, [lowcut_normalized, highcut_normalized], btype='band')

buffer_size = 512
eeg_buffer = np.zeros(buffer_size)
time_window = 4  
print_frequency = 2  
start_time = time.time()
predictions = []
while True:
    raw_data = ser.readline().decode("latin-1").strip()

    try:
        if raw_data:
            #eeg_value = raw_data.split(',')
            eeg_value = float(raw_data) 
            #print(eeg_value)
        else:
            print("No data received")    
        #decoded_data = raw_data.decode('utf-8').strip()
         # Assuming the data is a single float value

        # Update the buffer with the new value
        eeg_buffer[:-1] = eeg_buffer[1:]
        eeg_buffer[-1] = eeg_value
        #eeg_buffer = np.convolve(eeg_buffer, np.ones(5)/5, mode='valid')
        #print(eeg_buffer)
        # Apply notch filter
        eeg_buffer = signal.filtfilt(b_notch, a_notch, eeg_buffer)

        # Apply bandpass filter
        eeg_buffer = signal.filtfilt(b_bandpass, a_bandpass, eeg_buffer)

        # Feature extraction (similar to the training process)
        segment_features = calculate_psd_features(eeg_buffer,512)
        additional_features = calculate_additional_features(eeg_buffer, 512)
        segment_features = {**segment_features, **additional_features}

        features.append(segment_features)
        df = pd.DataFrame(features, columns=columns)
        X_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(X_scaled, columns=columns)
        # Reshape features for prediction
        input_data = np.array([list(df_scaled.iloc[-1].values)])
        probabilities = clf.predict_proba(input_data)
        # Check if the probability of class 1 is greater than the threshold
        #prediction = (probabilities[:, 1] > 0.4).astype(int)
        # Make real-time prediction
        prediction = clf.predict(input_data)
        # Print the classified class
        #if (prediction == 0):
        #    print("Relaxed")
        #else:
        #    print("Stressed")
        print(f"Predicted Class: {prediction}")
        print(probabilities)
        #print(eeg_value)
        #if prediction == 0:
        #    pyautogui.keyDown('space')                                                                 
        #elif prediction == 1:
        #    pyautogui.keyDown('w')
        time.sleep(.5)
    except UnicodeDecodeError as e:
        print(f"Error decoding data: {e}")
