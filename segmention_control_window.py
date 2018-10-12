import sys, math, pdb, collections
from minisom import MiniSom

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMenu, QSizePolicy, QMessageBox,  QCheckBox
from PyQt5.QtWidgets import  QDialog,\
    QHBoxLayout, QLabel, QGroupBox, QSlider, \
    QLineEdit, QScrollArea, QComboBox
from PyQt5.QtCore import Qt
import os, pywt
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random
from win32api import GetSystemMetrics

from scipy.signal import butter, lfilter, spectrogram, find_peaks, iirfilter
import  scipy.fftpack as fftpack
from skimage.feature import peak_local_max

class SegmentationControlWinow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QVBoxLayout()

        self.voltage_gain = 13.1070
        self.filter_low_pass = 2000
        self.filter_high_pass = 20
        self.filter_notch = [50]
        self.crop_data = 0  # 144843(6s in order to keep 5 seconds of data if fs=23437.5 and sample length = 262124)
        self.coeff_mav = 30
        self.coeff_thresh = 5
        self.min_peak_volt = 30
        self.max_peak_volt = 500 # Paper: 100
        self.segment_window = 6
        self.segment_amp_rise_duration = 1.0 # Paper: 0.1 (ms)
        self.segment_amp_rise_threshold = 40
        self.segment_threshold_calculated = 'N/A'
        self.save_preprocessed_data_file = "preprocessed_data"
        self.save_preprocessing_params_file = "preprocessing_params"
        self.save_peaks_file = "peaks"

        self.voltage_gain_label = QLabel('Voltage Gain(/uV): ', self)
        self.voltage_gain_input = QLineEdit(self)
        self.filter_low_pass_label = QLabel('Low Pass Filter(Hz): ', self)
        self.filter_low_pass_input = QLineEdit(self)
        self.filter_high_pass_label = QLabel('High Pass Filter(Hz): ', self)
        self.filter_high_pass_input = QLineEdit(self)
        self.filter_notch_label = QLabel('Notch Filter: ', self)
        self.filter_notch_input = QLineEdit(self)


        self.voltage_gain_widget = QGroupBox('Raw Signal Parameters', self)
        self.voltage_gain_widget_layout = QHBoxLayout()
        self.voltage_gain_widget.setLayout(self.voltage_gain_widget_layout)
        self.voltage_gain_widget_layout.addWidget(self.voltage_gain_label)
        self.voltage_gain_widget_layout.addWidget(self.voltage_gain_input)
        self.voltage_gain_widget_layout.addWidget(self.filter_low_pass_label)
        self.voltage_gain_widget_layout.addWidget(self.filter_low_pass_input)
        self.voltage_gain_widget_layout.addWidget(self.filter_high_pass_label)
        self.voltage_gain_widget_layout.addWidget(self.filter_high_pass_input)
        self.voltage_gain_widget_layout.addWidget(self.filter_notch_label)
        self.voltage_gain_widget_layout.addWidget(self.filter_notch_input)

        self.data_sample_crop_label = QLabel('Crop Data(+int):', self)
        self.data_sample_crop_input = QLineEdit(self)
        self.data_sample_crop_param = QComboBox(self)
        self.CROP_STYLES_NONE = 'None'
        self.CROP_STYLES_BOTH = 'From both sides'
        self.CROP_STYLE_START = 'From start'
        self.CROP_STYLE_END = 'From end'
        self.data_sample_crop_param.addItem(self.CROP_STYLES_NONE)
        self.data_sample_crop_param.addItem(self.CROP_STYLES_BOTH)
        self.data_sample_crop_param.addItem(self.CROP_STYLE_END)
        self.data_sample_crop_param.addItem(self.CROP_STYLE_START)
        self.data_sample_crop_widget = QGroupBox('Cropped Signal Parameters', self)
        self.data_sample_crop_widget_layout = QHBoxLayout()
        self.data_sample_crop_widget.setLayout(self.data_sample_crop_widget_layout)
        self.data_sample_crop_widget_layout.addWidget(self.data_sample_crop_label)
        self.data_sample_crop_widget_layout.addWidget(self.data_sample_crop_input)
        self.data_sample_crop_widget_layout.addWidget(self.data_sample_crop_param)

        self.peak_detection_threshold_mav_coeff_label = QLabel('Coefficient of Mean Absolute Value(+int):', self)
        self.peak_detection_threshold_mav_coeff_input = QLineEdit(self)
        self.peak_detection_threshold_output_coeff_label = QLabel('Coefficient of Peak detection threshold(+int):', self)
        self.peak_detection_threshold_output_coeff_input = QLineEdit(self)
        self.peak_detection_threshold_min_voltage_label = QLabel('Min Peak Voltage(uV)(+int): ', self)
        self.peak_detection_threshold_min_voltage_input = QLineEdit(self)
        self.peak_detection_threshold_max_voltage_label = QLabel('Max Peak Voltage(uV)(+int): ', self)
        self.peak_detection_threshold_max_voltage_input = QLineEdit(self)
        self.peak_detection_window_label = QLabel('Segmentation Window(ms)(+float): ')
        self.peak_detection_window_input = QLineEdit(self)
        self.segment_amp_rise_duration_label = QLabel('Peak Amplitude rise duration(ms)(+float): ', self)
        self.segment_amp_rise_duration_input = QLineEdit(self)
        self.segment_amp_rise_threshold_label = QLabel('Peak Amplitude rise threshold(uV)(+int): ', self)
        self.segment_amp_rise_threshold_input = QLineEdit(self)
        self.segment_threshold_calculated_label = QLabel('Minimum Peak Amplitude Threshold(uV): ', self)
        self.segment_threshold_calculated_input = QLabel('N/A', self)


        self.peak_detection_threshold_widget = QGroupBox('Peak detection Parameters', self)
        self.peak_detection_threshold_widget_layout = QHBoxLayout()
        self.peak_detection_threshold_widget.setLayout(self.peak_detection_threshold_widget_layout)
        self.peak_detection_threshold_left_widget = QGroupBox('', self)
        self.peak_detection_threshold_left_widget_layout = QVBoxLayout()
        self.peak_detection_threshold_left_widget.setLayout( self.peak_detection_threshold_left_widget_layout)
        self.peak_detection_threshold_right_widget = QGroupBox('', self)
        self.peak_detection_threshold_right_widget_layout = QVBoxLayout()
        self.peak_detection_threshold_right_widget.setLayout(self.peak_detection_threshold_right_widget_layout)
        self.peak_detection_threshold_widget_layout.addWidget(self.peak_detection_threshold_left_widget)
        self.peak_detection_threshold_widget_layout.addWidget(self.peak_detection_threshold_right_widget)


        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_mav_coeff_label)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_mav_coeff_input)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_output_coeff_label)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_output_coeff_input)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_min_voltage_label)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_min_voltage_input)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_max_voltage_label)
        self.peak_detection_threshold_left_widget_layout.addWidget(self.peak_detection_threshold_max_voltage_input)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.peak_detection_window_label)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.peak_detection_window_input)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.segment_amp_rise_duration_label)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.segment_amp_rise_duration_input)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.segment_amp_rise_threshold_label)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.segment_amp_rise_threshold_input)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.segment_threshold_calculated_label)
        self.peak_detection_threshold_right_widget_layout.addWidget(self.segment_threshold_calculated_input)

        self.save_preprocessed_data_checkbox = QCheckBox('Save Preprocessed data', self)
        self.save_peaks_checkbox = QCheckBox('Save Peaks', self)
        self.save_checkbox_widget = QGroupBox('Save Parameters', self)
        self.save_checkbox_widget_layout = QHBoxLayout()
        self.save_checkbox_widget.setLayout(self.save_checkbox_widget_layout)
        self.save_checkbox_widget_layout.addWidget(self.save_preprocessed_data_checkbox)
        self.save_checkbox_widget_layout.addWidget(self.save_peaks_checkbox)


        self.save_controls_button = QPushButton('Save parameters', self)
        self.save_controls_button.clicked.connect(self.save_params)

        self.layout.addWidget(self.voltage_gain_widget)
        self.layout.addWidget(self.data_sample_crop_widget)
        self.layout.addWidget(self.peak_detection_threshold_widget)
        self.layout.addWidget(self.save_checkbox_widget)
        self.layout.addWidget(self.save_controls_button)

        self.setLayout(self.layout)
        self.display_params()


    def reset_default(self):
        self.voltage_gain = 13.1070
        self.filter_low_pass = 2000
        self.filter_high_pass = 20
        self.filter_notch = [50]
        self.crop_data = 'None'
        self.coeff_mav = 30
        self.coeff_thresh = 5
        self.min_peak_volt = 30
        self.max_peak_volt = 100
        self.segment_window = 6
        self.segment_amp_rise_duration = 0.1
        self.segment_amp_rise_threshold = 40
        self.segment_threshold_calculated = 'N/A'




    def display_params(self):
        self.voltage_gain_input.setText(str(self.voltage_gain))
        self.filter_low_pass_input.setText(str(self.filter_low_pass))
        self.filter_high_pass_input.setText(str(self.filter_high_pass))
        if len(self.filter_notch)> 0:
            filter_notch = ""+str(self.filter_notch[0])
            for i in range(1, len(self.filter_notch)):
                filter_notch = filter_notch + "," + str(self.filter_notch[i])
        else:
            filter_notch = "50"
        self.filter_notch_input.setText(filter_notch)
        self.data_sample_crop_input.setText(str(self.crop_data))
        self.peak_detection_threshold_mav_coeff_input.setText(str(self.coeff_mav))
        self.peak_detection_threshold_output_coeff_input.setText(str(self.coeff_thresh))
        self.peak_detection_threshold_min_voltage_input.setText(str(self.min_peak_volt))
        self.peak_detection_threshold_max_voltage_input.setText(str(self.max_peak_volt))
        self.peak_detection_window_input.setText(str(self.segment_window))
        self.segment_amp_rise_duration_input.setText(str(self.segment_amp_rise_duration))
        self.segment_amp_rise_threshold_input.setText(str(self.segment_amp_rise_threshold))
        self.segment_threshold_calculated_input.setText(str(self.segment_threshold_calculated))

    def save_params(self):
        voltage_gain = float(self.voltage_gain_input.text())
        filter_low_pass = int(self.filter_low_pass_input.text())
        filter_high_pass = int(self.filter_high_pass_input.text())
        notches = self.filter_notch_input.text()
        notches = notches.split(",")
        notches_int = []
        if len(notches) > 0:
            for i in range(len(notches)):
                notches_int.append(int(notches[i].replace(" ", "")))
        else:
            notches_int = [50]
        filter_notch = notches_int
        crop_data = int(self.data_sample_crop_input.text())
        coeff_mav = int(self.peak_detection_threshold_mav_coeff_input.text())
        coeff_thresh = int(self.peak_detection_threshold_output_coeff_input.text())
        min_peak_volt = int(self.peak_detection_threshold_min_voltage_input.text())
        max_peak_volt = int(self.peak_detection_threshold_max_voltage_input.text())
        segment_window = float(self.peak_detection_window_input.text())
        segment_amp_rise_duration = float(self.segment_amp_rise_duration_input.text())
        segment_amp_rise_threshold = int(self.segment_amp_rise_threshold_input.text())

        if voltage_gain < 0:
            voltage_gain = 13.1070
        if filter_low_pass < 0:
            filter_low_pass = 2000
        if filter_high_pass < 0:
            filter_high_pass = 20
        if len(filter_notch) == 0:
            filter_notch = [50]
        if crop_data < 0:
            crop_data = 0
        if coeff_mav <= 0:
            coeff_mav = 30
        if coeff_thresh <= 0:
            coeff_thresh = 5
        if min_peak_volt < 0:
            min_peak_volt = 30
        if max_peak_volt <= 0:
            max_peak_volt = 100
        if segment_window <= 0:
            segment_window = 6
        if segment_amp_rise_duration <= 0:
            segment_amp_rise_duration = 0.1
        if segment_amp_rise_threshold <= 0:
            segment_amp_rise_threshold = 40

        self.voltage_gain = voltage_gain
        self.filter_low_pass = filter_low_pass
        self.filter_high_pass = filter_high_pass
        self.filter_notch = filter_notch
        self.crop_data = crop_data
        self.coeff_mav = coeff_mav
        self.coeff_thresh = coeff_thresh
        self.min_peak_volt = min_peak_volt
        self.max_peak_volt = max_peak_volt
        self.segment_window = segment_window
        self.segment_amp_rise_duration = segment_amp_rise_duration
        self.segment_amp_rise_threshold = segment_amp_rise_threshold
        self.display_params()
    def crop_input_data(self, data):
        crop_style = str(self.data_sample_crop_param.currentText())
        if crop_style != self.CROP_STYLES_NONE and self.crop_data > 0 and self.crop_data < len(data):
            if crop_style == self.CROP_STYLES_BOTH:
                crop_left = int(self.crop_data/2)
                crop_right = self.crop_data - crop_left
                return data[crop_left:len(data)-crop_right]
            elif crop_style == self.CROP_STYLE_START:
                return data[self.crop_data:len(data)]
            elif crop_style == self.CROP_STYLE_END:
                return data[:len(data)-self.crop_data]
        return data

    def butter_bandpass(self, cutoff_freqs, fs, btype, order=5):
        nyq = 0.5 * fs
        for i in range(len(cutoff_freqs)):
            cutoff_freqs[i] = cutoff_freqs[i] / nyq

        b, a = butter(order, cutoff_freqs, btype=btype)
        return b, a

    def butter_bandpass_filter(self, data, cutoff_freqs, btype, fs, order=5):
        b, a = self.butter_bandpass(cutoff_freqs.copy(), fs, btype, order=order)
        y = lfilter(b, a, np.copy(data))
        return y
    def Implement_Notch_Filter(self, time, data, band=20, freq=60, ripple=100, order=2, filter_type='butter'):
        fs = 1 / time
        nyq = fs / 2.0
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                         analog=False, ftype=filter_type)
        filtered_data = lfilter(b, a, data)
        return filtered_data
    def filter_data(self, data, fs):
        if self.filter_low_pass == 0 and self.filter_high_pass == 0:
            filtered = data
        elif self.filter_low_pass == 0:
            band_type = 'highpass'
            order=2
            filtered = self.butter_bandpass_filter(data, [self.filter_high_pass], band_type, fs, order)
        elif self.filter_high_pass == 0:
            band_type = 'lowpass'
            order = 2
            filtered = self.butter_bandpass_filter(data, [self.filter_low_pass], band_type, fs, order)
        else:
            band_type = 'band'
            order = 2
            filtered = self.butter_bandpass_filter(data, [self.filter_high_pass, self.filter_low_pass], band_type, fs, order)
        for i in range(len(self.filter_notch)):
            if self.filter_notch[i] > 0:
                filtered = self.Implement_Notch_Filter(1/fs, filtered, freq=self.filter_notch[i])
        return filtered
    def get_peak_threshold(self, data):
        max_data = np.amax(data)
        mav = np.sum(np.abs(data))/len(data)

        if max_data > self.coeff_mav * mav:
            thresh = self.coeff_thresh * mav
        else:
            thresh = max_data/self.coeff_thresh
        return thresh
    def get_muap_sample_length(self, fs):
        return int((fs * self.segment_window) / 1000)
    def get_muap_waveforms(self, peaks, data, fs):

        sample_length = self.get_muap_sample_length(fs)
        sample_left = int(sample_length/2)
        sample_right = sample_length - sample_left
        waveforms = []
        firing_time = []
        for i in range(len(peaks)):
            if peaks[i]-sample_left >= 0 and peaks[i] + sample_right <=len(data):
                waveforms.append(data[peaks[i]-sample_left:peaks[i] + sample_right])
                firing_time.append([peaks[i]])
        return waveforms, firing_time
    def detect_peaks(self, data, fs):
        threshold = self.get_peak_threshold(data)
        self.segment_threshold_calculated = str(int(threshold))
        self.segment_threshold_calculated_input.setText(self.segment_threshold_calculated)

        peak_range = [self.min_peak_volt, self.max_peak_volt]
        if threshold < self.min_peak_volt:
            peak_range[0] = threshold
        segment_window_samples = int((fs * self.segment_window) / 1000)

        peak_output = find_peaks(data, np.asarray(peak_range), distance=int(segment_window_samples/2))
        peaks = peak_output[0]
        return peaks

    def validate_peaks(self, data, fs, peaks):
        validated_peaks = []
        segment_amp_rise_duration_samples = int((fs * self.segment_amp_rise_duration) / 1000)

        for i in range(len(peaks)):
            if peaks[i]-segment_amp_rise_duration_samples >= 0:
               if data[peaks[i]] - np.amin(data[peaks[i]-segment_amp_rise_duration_samples: peaks[i]]) >= self.segment_amp_rise_threshold:
                   validated_peaks.append(peaks[i])


        return validated_peaks


