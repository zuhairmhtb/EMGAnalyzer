import sys, math, pdb, collections
from minisom import MiniSom
from PyQt5 import QtGui, QtCore
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

from MyNet.temp.emg_editor_pyqt.segmention_control_window import SegmentationControlWinow
from MyNet.temp.emg_editor_pyqt.classification_control_window import ClassificationControlWindow

class LoadingMessage(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QHBoxLayout()
        self.message = QLabel('Message', self)
        self.layout.addWidget(self.message)
        self.setLayout(self.layout)
    def enable(self, message=""):
        self.show()
        self.message.setText("Loading..." + str(message))
    def disable(self):
        self.hide()
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'EMG Editor'
        self.left = 10
        self.top = 10
        self.width = GetSystemMetrics(0)-100
        self.height = GetSystemMetrics(1)-100


        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)


        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_N:
            self.table_widget.display_next_data()

        elif event.key() == QtCore.Qt.Key_C:
            self.table_widget.classification_button_action()
        event.accept()



class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.parent = parent

        self.data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
        self.current_data = []
        self.output_motor_classes = 8
        self.current_peaks = []
        self.firing_table = [[] for _ in range(self.output_motor_classes)]
        self.plot_colors = [
            'brown', 'purple', 'orange', 'magenta', 'yellow', 'black', 'red', 'aqua', 'grey', 'olive',
            'wheat', 'cyan'
        ]


        self.data = self.get_dataset()
        self.preprocess_tab_fig = plt.figure(1)
        self.preprocess_tab_ax = []
        self.preprocess_tab_ax.append(self.preprocess_tab_fig.add_subplot(1, 1, 1))
        self.preprocess_tab_canvas = FigureCanvas(self.preprocess_tab_fig)
        self.preprocess_tab_toolbar = NavigationToolbar(self.preprocess_tab_canvas, self)

        self.segmentation_section_fig = plt.figure(2)
        self.segmentation_section_ax = []
        self.segmentation_section_ax.append(self.segmentation_section_fig.add_subplot(1, 1, 1))
        self.segmentation_section_ax[0].set_prop_cycle(color=self.plot_colors)
        self.segmentation_section_canvas = FigureCanvas(self.segmentation_section_fig)
        self.segmentation_section_toolbar = NavigationToolbar(self.segmentation_section_canvas, self)

        self.muap_waveforms_section_fig = plt.figure(3)
        self.muap_waveforms_section_ax = []
        self.muap_waveforms_section_ax.append(self.muap_waveforms_section_fig.add_subplot(1, 1, 1))
        self.muap_waveforms_section_canvas = FigureCanvas(self.muap_waveforms_section_fig)
        self.muap_waveforms_section_toolbar = NavigationToolbar(self.muap_waveforms_section_canvas, self)

        self.firing_table_section_fig = plt.figure(4)
        self.firing_table_section_ax = []
        self.firing_table_section_ax.append(self.firing_table_section_fig.add_subplot(1, 1, 1))
        self.firing_table_section_ax[0].set_prop_cycle(color=self.plot_colors)
        self.firing_table_section_canvas = FigureCanvas(self.firing_table_section_fig)
        self.firing_table_section_toolbar = NavigationToolbar(self.firing_table_section_canvas, self)
        self.firing_table_section_fig.subplots_adjust(wspace=0.1, hspace=1)

        self.classification_section_fig = plt.figure(5)

        self.classification_section_ax = [
            self.classification_section_fig.add_subplot(2, self.output_motor_classes, i+1) for i in range(self.output_motor_classes*2)
        ]

        for i in range(len(self.classification_section_ax)):
            self.classification_section_ax[i].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False
            )
        self.classification_section_canvas = FigureCanvas(self.classification_section_fig)
        self.classification_section_toolbar = NavigationToolbar(self.classification_section_canvas, self)


        data = np.asarray([0 for _ in range(100)])
        self.preprocess_tab_ax[0].plot(data)
        self.preprocess_tab_ax[0].grid()
        self.preprocess_tab_canvas.draw()

        self.segmentation_section_ax[0].plot(data)
        self.segmentation_section_ax[0].grid()
        self.segmentation_section_canvas.draw()

        self.muap_waveforms_section_ax[0].plot(data)
        self.muap_waveforms_section_ax[0].grid()
        self.muap_waveforms_section_canvas.draw()

        self.firing_table_section_ax[0].plot(data)
        self.firing_table_section_ax[0].grid()
        self.firing_table_section_canvas.draw()

        for i in range(len(self.classification_section_ax)):
            self.classification_section_ax[i].plot(data)
            self.classification_section_ax[i].grid()
        self.classification_section_canvas.draw()

        self.segmentation_control_window = SegmentationControlWinow()
        self.classification_control_window = ClassificationControlWindow()
        self.loading_window = LoadingMessage()


        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "PreProcessing Tab")
        self.tabs.addTab(self.tab2, "MUAP Analysis Tab")


        # Create first tab
        self.initPreProcessingUI()
        self.tab1.setLayout(self.preprocessing_tab_layout)

        # Create Second tab
        self.initMUAPAnalysisUI()
        self.tab2.setLayout(self.muap_an_tab_layout)



        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


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

    def calculate_dwt(self, data, method='haar', thresholding='soft', level=1):
        print("Calculating Discrete Wavelet Transform")
        if level <= 1:
            (ca, cd) = pywt.dwt(data, method)
            cat = pywt.threshold(ca, np.std(ca) / 2, thresholding)
            cdt = pywt.threshold(cd, np.std(cd) / 2, thresholding)
            return cat, cdt
        else:
            decs = pywt.wavedec(data, method, level=level)
            result = []
            for d in decs:
                result.append(pywt.threshold(d, np.std(d) / 2, thresholding))
            return result
    def calculate_fourier_transform(self, np_data, fs, window=-1, overlap=0):
        if window <= 0 or window > len(np_data):
            fft = fftpack.fft(np_data, np_data.shape[0])
            fftfreq = fftpack.fftfreq(np_data.shape[0], fs)
            return fftfreq, None, fft
        else:
            f, t, fft = spectrogram(np_data, fs, window=('hamming'), noverlap=overlap, return_onesided=True,
                                    mode='complex', nperseg=window)
            return f, t, fft

    # Required input defintions are as follows;
    # time:   Time between samples
    # band:   The bandwidth around the centerline freqency that you wish to filter
    # freq:   The centerline frequency to be filtered
    # ripple: The maximum passband ripple that is allowed in db
    # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
    #         IIR filters are best suited for high values of order.  This algorithm
    #         is hard coded to FIR filters
    # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    # data:         the data to be filtered
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
    def perform_muap_analysis(self, data, fs):
        # Signal Filtering
        pass_type = 'lowpass'
        pass_range=[8000]  # 8KHz
        filtered = self.butter_bandpass_filter(data, pass_range, pass_type, fs)

        return filtered
    def read_sampling_rate(self, path):
        print("Reading sampling rate from", path)
        file = open(path, 'r')
        print("Content:")
        content = file.read().split("\n")
        sampling_rate = float(content[0].split(" ")[2])
        print(str(sampling_rate))
        return sampling_rate

    def update_firing_table(self, firing_time, output_class, output_superimposition, append=True):
        if not append:
            for i in range(len(output_class)):
                if output_superimposition[i] == 0:
                    self.firing_table[output_class[i]]= []
        for i in range(len(firing_time)):
            if output_superimposition[i] == 0:
                self.firing_table[output_class[i]] += firing_time[i]

        print(self.firing_table)

    def get_dataset(self):
        data_type = {}
        for dt in os.listdir(self.data_base_dir):
            dt_p = os.path.join(self.data_base_dir, dt)
            if os.path.isdir(dt_p):
                data_type[dt] = {}
                for disease in os.listdir(dt_p):
                    disease_p = os.path.join(dt_p, disease)
                    if os.path.isdir(disease_p):
                        data_type[dt][disease] ={}
                        for pat in os.listdir(disease_p):
                            pat_p = os.path.join(disease_p, pat)
                            if os.path.isdir(pat_p):
                                data_type[dt][disease][pat] = {}
                                for rec in os.listdir(pat_p):
                                    rec_p = os.path.join(pat_p, rec)
                                    if os.path.isdir(rec_p):
                                        data_type[dt][disease][pat][rec] = rec_p

        print(data_type)
        return data_type
    def get_current_data_path(self):
        return self.data[str(self.data_type_widget.currentText())][str(self.disease_type_widget.currentText())][str(self.patient_id_widget.currentText())][str(self.record_widget.currentText())]
    def get_data(self, d_path):
        np_data = np.load(d_path).flatten()
        if self.filter_activate_box.isChecked():
            filter_band = str(self.filter_pass_type_input.currentText()).lower()
            filter_order = int(self.filter_order_input.currentText())
            filter_range_int = [20, 600]
            filter_range = str(self.filter_pass_range_input.text())
            if filter_band == 'band':
                if len(filter_range) != 0:
                    filter_range = filter_range.split(",")
                    if len(filter_range) == 2:
                        filter_range_int = [int(filter_range[i].replace(" ", "")) for i in range(len(filter_range))]
            elif filter_band == 'lowpass':
                if len(filter_range) > 0:
                    filter_range_int = [int(filter_range)]
                else:
                    filter_range_int = [600]
            elif filter_band == 'highpass':
                if len(filter_range) > 0:
                    filter_range_int = [int(filter_range)]
                else:
                    filter_range_int = [100]
            if os.path.exists(os.path.join(self.get_current_data_path(), 'data.hea')):
                fs = self.read_sampling_rate(os.path.join(self.get_current_data_path(), 'data.hea'))
            else:
                fs = 44100
            np_data = self.butter_bandpass_filter(np_data, filter_range_int, filter_band, fs, order=filter_order)

        if self.dwt_activate_box.isChecked():
            print('Calculating DWT')
            method = str(self.dwt_method_input.text()).lower()
            thresh = str(self.dwt_thresholding_input.currentText()).lower()
            level = int(self.dwt_level_input.currentText())
            output_index = int(self.dwt_output.currentText())

            dwt_result = self.calculate_dwt(np_data, method=method, thresholding=thresh, level=level)

            if output_index > level:
                output_index = 0
                self.dwt_output.setCurrentIndex(0)
            np_data = dwt_result[output_index]
        return np_data
    def set_data_type_action(self):
        data_type = os.path.join(self.data_base_dir, str(self.data_type_widget.currentText()))
        self.disease_type_widget.clear()
        for disease in os.listdir(data_type):
            disease_p = os.path.join(data_type, disease)
            if os.path.isdir(disease_p):
                self.disease_type_widget.addItem(disease)
        disease = os.path.join(data_type, str(self.disease_type_widget.currentText()))
        self.patient_id_widget.clear()
        for pat in os.listdir(disease):
            pat_p = os.path.join(disease, pat)
            if os.path.isdir(pat_p):
                self.patient_id_widget.addItem(pat)
        patient = os.path.join(disease, str(self.patient_id_widget.currentText()))
        self.record_widget.clear()
        for rec in os.listdir(patient):
            rec_p = os.path.join(patient, rec)
            if os.path.isdir(rec_p):
                self.record_widget.addItem(rec)
        self.set_record_type_action()
    def set_disease_type_action(self):
        disease = os.path.join(os.path.join(self.data_base_dir,
                               str(self.data_type_widget.currentText())), str(self.disease_type_widget.currentText()))
        self.patient_id_widget.clear()
        for pat in os.listdir(disease):
            pat_p = os.path.join(disease, pat)
            if os.path.isdir(pat_p):
                self.patient_id_widget.addItem(pat)
        patient = os.path.join(disease, str(self.patient_id_widget.currentText()))
        self.record_widget.clear()
        for rec in os.listdir(patient):
            rec_p = os.path.join(patient, rec)
            if os.path.isdir(rec_p):
                self.record_widget.addItem(rec)
        self.set_record_type_action()
    def set_patient_type_action(self):
        patient = os.path.join(os.path.join(os.path.join(self.data_base_dir, str(self.data_type_widget.currentText())),
                                            str(self.disease_type_widget.currentText())),
                               str(self.patient_id_widget.currentText()))
        self.record_widget.clear()
        for rec in os.listdir(patient):
            rec_p = os.path.join(patient, rec)
            if os.path.isdir(rec_p):
                self.record_widget.addItem(rec)
        self.set_record_type_action()
    def set_record_type_action(self):
        """data = str(self.data_type_widget.currentText()) + '\\' + str(self.disease_type_widget.currentText()) + \
               '\\' + str(self.patient_id_widget.currentText()) + '\\' + str(self.record_widget.currentText()) + '\\' + 'data.npy'
        data_p = os.path.join(self.data_base_dir, data)"""

        data_type = os.path.join(self.data_base_dir, str(self.data_type_widget.currentText()))
        disease_type = os.path.join(data_type, str(self.disease_type_widget.currentText()))
        pat_type = os.path.join(disease_type, str(self.patient_id_widget.currentText()))
        rec_type = os.path.join(pat_type, str(self.record_widget.currentText()))
        data_p = os.path.join(rec_type, 'data.npy')
        print('Loading data from ' + str(data_p))
        if os.path.exists(data_p):
            self.current_data = self.get_data(data_p)
            self.segmentation_section_ax[0].clear()
            self.segmentation_section_ax[0].plot(self.current_data, 'b-')
            self.segmentation_section_ax[0].grid()
            self.segmentation_section_canvas.draw()

            self.peak_activate_button_action()
    def display_next_data(self):
        cur_rec_ind = self.record_widget.currentIndex()
        tot_rec = self.record_widget.count()
        if cur_rec_ind < tot_rec-1:
            self.record_widget.setCurrentIndex(cur_rec_ind+1)
            self.set_record_type_action()
        else:
            cur_pat_index = self.patient_id_widget.currentIndex()
            tot_pat = self.patient_id_widget.count()
            if cur_pat_index < tot_pat - 1:
                self.patient_id_widget.setCurrentIndex(cur_pat_index+1)
                self.set_patient_type_action()
            else:
                cur_dis_ind = self.disease_type_widget.currentIndex()
                tot_dis = self.disease_type_widget.count()
                if cur_dis_ind < tot_dis - 1:
                    self.disease_type_widget.setCurrentIndex(cur_dis_ind+1)
                    self.set_disease_type_action()
                else:
                    cur_dat_ind = self.data_type_widget.currentIndex()
                    tot_dat = self.data_type_widget.count()
                    if cur_dis_ind < tot_dat - 1:
                        self.data_type_widget.setCurrentIndex(cur_dat_ind+1)
                    else:
                        self.data_type_widget.setCurrentIndex(0)
                    self.set_data_type_action()

    def graph_info_button_action(self):

        info_path = os.path.join(self.get_current_data_path(), 'data.hea')

        msg = 'No Information Found for ' + str(info_path)
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                msg = f.read()
        buttonReply = QMessageBox.question(self, 'PyQt5 message', msg,
                                           QMessageBox.Ok, QMessageBox.Ok)
        if buttonReply == QMessageBox.Ok:
            print('Yes clicked.')
        else:
            print('No clicked.')
    def fft_activate_button_action(self):
        if self.fft_activate_box.isChecked():
            window = str(self.fft_window.text())
            overlap = str(self.fft_overlap.text())
            if len(window) == 0:
                window = 0
            else:
                window = int(window)
            if len(overlap) == 0:
                overlap = 0
            else:
                overlap = int(overlap)
            fs = self.read_sampling_rate(os.path.join(self.get_current_data_path(), 'data.hea'))

            f, t, fft = self.calculate_fourier_transform(self.current_data, fs, window=window, overlap=overlap)

            self.preprocess_tab_ax[0].clear()
            if not (t is None):
                self.preprocess_tab_ax[0].pcolormesh(t, f, np.abs(fft)**2, cmap='viridis')
                self.preprocess_tab_ax[0].set_xlabel('Time Bins[n]')
                self.preprocess_tab_ax[0].set_ylabel('Frequency[Hz]')
            else:
                self.preprocess_tab_ax[0].plot(f, np.abs(fft)**2)
                self.preprocess_tab_ax[0].set_xlabel('Frequency [Hz]')
                self.preprocess_tab_ax[0].set_ylabel('Energy')
            self.preprocess_tab_ax[0].grid()
            self.preprocess_tab_canvas.draw()
        else:
            self.preprocess_tab_ax[0].clear()
            self.preprocess_tab_ax[0].plot(self.current_data)
            self.preprocess_tab_ax[0].grid()
            self.preprocess_tab_ax[0].set_xlabel('Samples[n]')
            self.preprocess_tab_ax[0].set_ylabel('Voltage[uV]')
            self.preprocess_tab_canvas.draw()

    def peak_activate_button_action(self):
        self.preprocess_tab_ax[0].clear()
        self.preprocess_tab_ax[0].grid()
        self.preprocess_tab_ax[0].set_xlabel('Samples[n]')
        self.preprocess_tab_ax[0].set_ylabel(' Voltage[uV]')
        self.preprocess_tab_ax[0].plot(self.current_data)
        data = self.current_data.copy()
        if self.peak_invert_box.isChecked():
            data = data * (-1)
        if self.peak_activate_box.isChecked():
            thresh_min = np.mean(np.abs(data))
            thresh_max = np.max(data)
            thresh_per = int(self.peak_height.value())
            height = thresh_min + int(((thresh_max-thresh_min)*thresh_per)/100)
            peaks, props = find_peaks(data, height=height)
            print('Peak Minimum Threshold: ' + str(thresh_per) + '% greater than mean amplitude')
            self.peak_total_label.setText(str(len(peaks)))
            self.preprocess_tab_ax[0].plot(peaks, self.current_data[peaks], 'x')
        self.preprocess_tab_canvas.draw()

    def initPreProcessingUI(self):
        self.preprocessing_tab_layout = QVBoxLayout()
        self.plot_widget = QGroupBox('EMG Graph')
        self.plot_widget_layout = QHBoxLayout()
        self.button_widget = QGroupBox('Buttons')
        self.button_widget_layout = QHBoxLayout()
        self.file_widget = QGroupBox('File Information')
        self.file_widget_layout = QHBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget(self.scroll_area)
        self.scroll_area_layout = QVBoxLayout()
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.preprocessing_widget = QGroupBox('Pre-Processing')
        self.preprocessing_widget_layout = QHBoxLayout()
        self.scroll_area_layout.addWidget(self.preprocessing_widget)

        self.data_type_widget = QComboBox(self)

        self.data_type_widget.activated.connect(self.set_data_type_action)
        self.disease_type_widget = QComboBox(self)
        self.disease_type_widget.activated.connect(self.set_disease_type_action)
        self.patient_id_widget = QComboBox(self)
        self.patient_id_widget.activated.connect(self.set_patient_type_action)
        self.record_widget = QComboBox(self)
        self.record_widget.activated.connect(self.set_record_type_action)
        self.graph_info_button = QPushButton('Information', self)
        self.graph_info_button.clicked.connect(self.graph_info_button_action)

        self.filter_pass_type_input = QComboBox(self)
        self.filter_pass_type_input.addItem('Band')
        self.filter_pass_type_input.addItem('Lowpass')
        self.filter_pass_type_input.addItem('Highpass')
        self.filter_pass_type_input_widget = QGroupBox('Filter Pass', self)
        self.filter_pass_type_input_widget_layout = QHBoxLayout()
        self.filter_pass_type_input_widget.setLayout(self.filter_pass_type_input_widget_layout)
        self.filter_pass_type_input_widget_layout.addWidget(self.filter_pass_type_input)

        self.filter_order_input = QComboBox(self)
        self.filter_order_input.addItem('1')
        self.filter_order_input.addItem('2')
        self.filter_order_input.addItem('3')
        self.filter_order_input.addItem('4')
        self.filter_order_input.addItem('5')
        self.filter_order_input_widget = QGroupBox('Filter Order', self)
        self.filter_order_input_widget_layout = QHBoxLayout()
        self.filter_order_input_widget.setLayout(self.filter_order_input_widget_layout)
        self.filter_order_input_widget_layout.addWidget(self.filter_order_input)

        self.filter_pass_range_input = QLineEdit(self)
        self.filter_pass_range_input_widget = QGroupBox('Filter Range', self)
        self.filter_pass_range_input_widget_layout = QHBoxLayout()
        self.filter_pass_range_input_widget.setLayout(self.filter_pass_range_input_widget_layout)
        self.filter_pass_range_input_widget_layout.addWidget(self.filter_pass_range_input)

        self.filter_activate_box = QCheckBox('Activate', self)
        self.buffer_filter_widget = QGroupBox('Buffer Filter')
        self.buffer_filter_widget_layout = QVBoxLayout()

        self.dwt_method_input = QLineEdit(self)
        self.dwt_method_input_widget = QGroupBox('Mother Wavelet', self)
        self.dwt_method_input_widget_layout = QHBoxLayout()
        self.dwt_method_input_widget.setLayout(self.dwt_method_input_widget_layout)
        self.dwt_method_input_widget_layout.addWidget(self.dwt_method_input)

        self.dwt_thresholding_input = QComboBox(self)
        self.dwt_thresholding_input.addItem('Soft')
        self.dwt_thresholding_input_widget = QGroupBox('Threshold', self)
        self.dwt_thresholding_input_widget_layout = QHBoxLayout()
        self.dwt_thresholding_input_widget.setLayout(self.dwt_thresholding_input_widget_layout)
        self.dwt_thresholding_input_widget_layout.addWidget(self.dwt_thresholding_input)

        self.dwt_level_input = QComboBox(self)
        self.dwt_level_input_widget = QGroupBox('Level', self)
        self.dwt_level_input_widget_layout = QHBoxLayout()
        self.dwt_level_input_widget.setLayout(self.dwt_level_input_widget_layout)
        self.dwt_level_input_widget_layout.addWidget(self.dwt_level_input)

        self.dwt_output = QComboBox(self)
        self.dwt_output.addItem('0')
        for i in range(10):
            self.dwt_level_input.addItem(str(i + 1))
            self.dwt_output.addItem(str(i + 1))
        self.dwt_output_widget = QGroupBox('Output Wavelet', self)
        self.dwt_output_widget_layout = QHBoxLayout()
        self.dwt_output_widget.setLayout(self.dwt_output_widget_layout)
        self.dwt_output_widget_layout.addWidget(self.dwt_output)

        self.dwt_activate_box = QCheckBox('Activate', self)
        self.dwt_widget = QGroupBox('Discrete Wavelet Transform')
        self.dwt_widget_layout = QVBoxLayout()

        self.fft_window = QLineEdit(self)
        self.fft_window_widget = QGroupBox('Window Size', self)
        self.fft_window_widget_layout = QHBoxLayout()
        self.fft_window_widget.setLayout(self.fft_window_widget_layout)
        self.fft_window_widget_layout.addWidget(self.fft_window)

        self.fft_overlap = QLineEdit(self)
        self.fft_overlap_widget = QGroupBox('Overlap', self)
        self.fft_overlap_widget_layout = QHBoxLayout()
        self.fft_overlap_widget.setLayout(self.fft_overlap_widget_layout)
        self.fft_overlap_widget_layout.addWidget(self.fft_overlap)

        self.fft_activate_box = QCheckBox('Show', self)
        self.fft_activate_box.clicked.connect(self.fft_activate_button_action)
        self.fft_widget = QGroupBox('Fast Fourier Transform', self)
        self.fft_widget_layout = QVBoxLayout()

        self.peak_height = QSlider(Qt.Horizontal)
        self.peak_height.setMinimum(0)
        self.peak_height.setMaximum(100)
        self.peak_height.setTickInterval(1)
        self.peak_height.setValue(0)
        self.peak_height.valueChanged.connect(self.peak_activate_button_action)

        self.peak_height_widget = QGroupBox('Peak Height', self)
        self.peak_height_widget_layout = QHBoxLayout()
        self.peak_height_widget.setLayout(self.peak_height_widget_layout)
        self.peak_height_widget_layout.addWidget(self.peak_height)

        self.peak_total_label = QLabel('N/A', self)
        self.peak_total_widget = QGroupBox('Total', self)
        self.peak_total_widget_layout = QHBoxLayout()
        self.peak_total_widget.setLayout(self.peak_total_widget_layout)
        self.peak_total_widget_layout.addWidget(self.peak_total_label)

        self.peak_activate_box = QCheckBox('Show', self)
        self.peak_invert_box = QCheckBox('Inverse', self)
        self.peak_activate_box.clicked.connect(self.peak_activate_button_action)
        self.peak_invert_box.clicked.connect(self.peak_activate_button_action)
        self.peak_widget = QGroupBox('Signal Peak')
        self.peak_widget_layout = QHBoxLayout()

        for dt in os.listdir(self.data_base_dir):
            dt_p = os.path.join(self.data_base_dir, dt)
            if os.path.isdir(dt_p):
                self.data_type_widget.addItem(dt)
        data_type = os.path.join(self.data_base_dir, str(self.data_type_widget.currentText()))
        for disease in os.listdir(data_type):
            disease_p = os.path.join(data_type, disease)
            if os.path.isdir(disease_p):
                self.disease_type_widget.addItem(disease)
        disease = os.path.join(data_type, str(self.disease_type_widget.currentText()))
        for pat in os.listdir(disease):
            pat_p = os.path.join(disease, pat)
            if os.path.isdir(pat_p):
                self.patient_id_widget.addItem(pat)
        patient = os.path.join(disease, str(self.patient_id_widget.currentText()))
        for rec in os.listdir(patient):
            rec_p = os.path.join(patient, rec)
            if os.path.isdir(rec_p):
                self.record_widget.addItem(rec)

        self.plot_widget_layout.addWidget(self.preprocess_tab_canvas)
        self.button_widget_layout.addWidget(self.preprocess_tab_toolbar)
        self.plot_widget.setLayout(self.plot_widget_layout)
        self.button_widget.setLayout(self.button_widget_layout)

        self.file_widget_layout.addWidget(self.data_type_widget)
        self.file_widget_layout.addWidget(self.disease_type_widget)
        self.file_widget_layout.addWidget(self.patient_id_widget)
        self.file_widget_layout.addWidget(self.record_widget)
        self.file_widget_layout.addWidget(self.graph_info_button)
        self.file_widget.setLayout(self.file_widget_layout)

        self.buffer_filter_widget_layout.addWidget(self.filter_pass_type_input_widget)
        self.buffer_filter_widget_layout.addWidget(self.filter_pass_range_input_widget)
        self.buffer_filter_widget_layout.addWidget(self.filter_order_input_widget)
        self.buffer_filter_widget_layout.addWidget(self.filter_activate_box)
        self.preprocessing_widget_layout.addWidget(self.buffer_filter_widget)
        self.preprocessing_widget.setLayout(self.preprocessing_widget_layout)
        self.buffer_filter_widget.setLayout(self.buffer_filter_widget_layout)

        self.dwt_widget_layout.addWidget(self.dwt_method_input_widget)
        self.dwt_widget_layout.addWidget(self.dwt_thresholding_input_widget)
        self.dwt_widget_layout.addWidget(self.dwt_level_input_widget)
        self.dwt_widget_layout.addWidget(self.dwt_output_widget)
        self.dwt_widget_layout.addWidget(self.dwt_activate_box)
        self.dwt_widget.setLayout(self.dwt_widget_layout)
        self.preprocessing_widget_layout.addWidget(self.dwt_widget)

        self.fft_widget_layout.addWidget(self.fft_window_widget)
        self.fft_widget_layout.addWidget(self.fft_overlap_widget)
        self.fft_widget_layout.addWidget(self.fft_activate_box)
        self.fft_widget.setLayout(self.fft_widget_layout)
        self.preprocessing_widget_layout.addWidget(self.fft_widget)

        self.peak_widget_layout.addWidget(self.peak_height_widget)
        self.peak_widget_layout.addWidget(self.peak_total_widget)
        self.peak_widget_layout.addWidget(self.peak_activate_box)
        self.peak_widget_layout.addWidget(self.peak_invert_box)
        self.peak_widget.setLayout(self.peak_widget_layout)
        self.button_widget_layout.addWidget(self.peak_widget)

        self.preprocessing_tab_layout.addWidget(self.file_widget)
        self.preprocessing_tab_layout.addWidget(self.plot_widget)
        self.preprocessing_tab_layout.addWidget(self.button_widget)
        self.preprocessing_tab_layout.addWidget(self.scroll_area)

    def segmentation_control_button_action(self):
        self.segmentation_control_window.segment_threshold_calculated = str(
           int( self.segmentation_control_window.get_peak_threshold(self.current_data))
        )
        self.segmentation_control_window.segment_threshold_calculated_input.setText(
            self.segmentation_control_window.segment_threshold_calculated
        )
        self.segmentation_control_window.show()
    def segmentation_detect_peak_button_action(self):
        fs = self.read_sampling_rate(os.path.join(self.get_current_data_path(), 'data.hea'))
        cropped_data = self.segmentation_control_window.crop_input_data(self.current_data)
        filtered = self.segmentation_control_window.filter_data(cropped_data,fs)
        if self.segmentation_control_window.save_preprocessed_data_checkbox.isChecked():
            np.save(os.path.join(self.get_current_data_path(), self.segmentation_control_window.save_preprocessed_data_file), filtered)
        peaks = self.segmentation_control_window.detect_peaks(
            filtered, fs
        )
        if self.segmentation_control_window.save_peaks_checkbox.isChecked():
            np.save(os.path.join(self.get_current_data_path(), self.segmentation_control_window.save_peaks_file), peaks)



        self.segmentation_section_ax[0].clear()
        self.segmentation_section_ax[0].plot(filtered, 'b-')
        self.segmentation_section_ax[0].plot(peaks, filtered[peaks], 'gx')
        self.segmentation_section_ax[0].grid()
        self.segmentation_section_ax[0].set_title('Total Peaks: ' + str(len(peaks)))
        self.segmentation_section_canvas.draw()

    def segmentation_validate_peak_button_action(self):
        fs = self.read_sampling_rate(os.path.join(self.get_current_data_path(), 'data.hea'))
        cropped_data = self.segmentation_control_window.crop_input_data(self.current_data)
        filtered = self.segmentation_control_window.filter_data(cropped_data, fs)
        if self.segmentation_control_window.save_preprocessed_data_checkbox.isChecked():
            np.save(os.path.join(self.get_current_data_path(), self.segmentation_control_window.save_preprocessed_data_file), filtered)
        peaks = self.segmentation_control_window.detect_peaks(filtered, fs)
        validated_peaks = self.segmentation_control_window.validate_peaks(filtered, fs, peaks)
        if self.segmentation_control_window.save_peaks_checkbox.isChecked():
            np.save(os.path.join(self.get_current_data_path(), self.segmentation_control_window.save_peaks_file), validated_peaks)

        self.segmentation_section_ax[0].clear()
        self.segmentation_section_ax[0].plot(filtered, 'b-')
        self.segmentation_section_ax[0].plot(validated_peaks, filtered[validated_peaks], 'gx')
        self.segmentation_section_ax[0].grid()
        self.segmentation_section_ax[0].set_title('Total Peaks: ' + str(len(validated_peaks)))
        #pdb.set_trace()

        self.segmentation_section_canvas.draw()

    def classification_button_action(self):
        self.firing_table = [[] for _ in range(self.output_motor_classes)]
        fs = self.read_sampling_rate(os.path.join(self.get_current_data_path(), 'data.hea'))
        cropped_data = self.segmentation_control_window.crop_input_data(self.current_data)
        filtered = self.segmentation_control_window.filter_data(cropped_data, fs)
        if self.segmentation_control_window.save_preprocessed_data_checkbox.isChecked():
            np.save(os.path.join(self.get_current_data_path(), self.segmentation_control_window.save_preprocessed_data_file), filtered)
        peaks = self.segmentation_control_window.detect_peaks(filtered, fs)
        validated_peaks = self.segmentation_control_window.validate_peaks(filtered, fs, peaks)
        if self.segmentation_control_window.save_peaks_checkbox.isChecked():
            np.save(os.path.join(self.get_current_data_path(), self.segmentation_control_window.save_peaks_file), validated_peaks)
        waveforms, firing_time = self.segmentation_control_window.get_muap_waveforms(validated_peaks, filtered, fs)

        if len(waveforms) > 0:
            weights = self.classification_control_window.get_ann_weights(len(waveforms[0]), self.output_motor_classes, waveforms)
            if not self.classification_control_window.perform_sofm_learning_checkbox.isChecked():
                weights = self.classification_control_window.sofm_learning_phase(waveforms, len(waveforms[0]), self.output_motor_classes, weights)
            if not self.classification_control_window.perform_lvq_learning_checkbox.isChecked():
                weights = self.classification_control_window.lvq_learning_phase(waveforms, len(waveforms[0]), self.output_motor_classes, weights)
            waveform_classes, waveform_superimposition = self.classification_control_window.sofm_classification(
                waveforms, len(waveforms[0]), self.output_motor_classes, weights
            )

            if not self.classification_control_window.perform_muap_class_averaging_checkbox.isChecked():
                waveforms, waveform_classes, waveform_superimposition, firing_time = self.classification_control_window.average_muap_templates(
                    waveforms, waveform_classes, waveform_superimposition, weights, len(waveforms[0]), self.output_motor_classes,
                    firing_time=firing_time
                )
            self.update_firing_table(firing_time, waveform_classes, waveform_superimposition, append=True)

            if self.classification_control_window.perform_muap_decomposition_checkbox.isChecked():
                self.loading_window.enable("Performing Decomposition of Superimposed Signals")
                updated_muaps, residue_superimposed = self.classification_control_window.perform_emg_decomposition(
                    waveforms, waveform_classes, waveform_superimposition, firing_time
                )
                self.loading_window.disable()
                print('Decomposition Complete')
                ft = []
                classes = []
                superimposition = [0] * len(updated_muaps)
                for i in range(len(updated_muaps)):
                    ft.append(updated_muaps[i][2])
                    classes.append(updated_muaps[i][1])
                self.update_firing_table(ft, classes, superimposition, append=True)
            if self.classification_control_window.save_muap_waveform_checkbox.isChecked():
                np.save(os.path.join(self.get_current_data_path(), self.classification_control_window.save_muap_waveform_file), waveforms)
            if self.classification_control_window.save_muap_classes_checkbox.isChecked():
                np.save(os.path.join(self.get_current_data_path(), self.classification_control_window.save_muap_classes_file), waveform_classes)
            if self.classification_control_window.save_muap_superimposition_checkbox.isChecked():
                np.save(os.path.join(self.get_current_data_path(), self.classification_control_window.save_muap_superimposition_file), waveform_superimposition)
            if self.classification_control_window.save_muap_firing_time_checkbox.isChecked():
                np.save(os.path.join(self.get_current_data_path(), self.classification_control_window.save_muap_firing_time_file), firing_time)
            if self.classification_control_window.save_muap_firing_table_checkbox.isChecked():
                np.save(os.path.join(self.get_current_data_path(), self.classification_control_window.save_muap_firing_table_file), self.firing_table)

            waveforms_detected = {i: 0 for i in range(self.output_motor_classes)}
            superimposition_detected = {i: 0 for i in range(self.output_motor_classes)}


            for i in range(self.output_motor_classes):
                self.classification_section_ax[i].clear()
                self.classification_section_ax[i+self.output_motor_classes].clear()

            for i in range(len(waveforms)):
                if waveform_superimposition[i] == 0:
                    self.classification_section_ax[waveform_classes[i]].plot(waveforms[i])
                    waveforms_detected[waveform_classes[i]] += 1
                else:
                    self.classification_section_ax[waveform_classes[i]+self.output_motor_classes].plot(waveforms[i])
                    superimposition_detected[waveform_classes[i]] += 1

            for i in range(self.output_motor_classes):
                self.classification_section_ax[i].grid()
                self.classification_section_ax[i].set_title('A:' + str(waveforms_detected[i]))
                self.classification_section_ax[i + self.output_motor_classes].set_title(
                    'S:' + str(superimposition_detected[i]))

            self.classification_section_canvas.draw()
            self.firing_table_section_ax[0].clear()
            self.firing_table_section_ax[0].grid()
            self.firing_table_section_ax[0].set_xlabel('Firing Time')
            self.firing_table_section_ax[0].set_ylabel('Motor Unit')
            self.segmentation_section_ax[0].clear()
            self.segmentation_section_ax[0].plot(filtered, 'b-')
            for i in range(len(self.firing_table)):
                if len(self.firing_table[i])> 0:
                    print('Motor unit: ' + str(i+1))
                    print('Firing time: ' + str(self.firing_table[i]))
                    print('Total fires: ' + str(len(self.firing_table[i])))
                    print('Total data points: ' + str(len(filtered)))

                    self.firing_table_section_ax[0].plot( (np.asarray(self.firing_table[i])*1000)/fs, [i+1]*len(self.firing_table[i]), 'x')
                    self.segmentation_section_ax[0].plot(self.firing_table[i], np.asarray(filtered)[self.firing_table[i]], 'o')

            for i in range(len(self.firing_table)):
                if len(self.firing_table[i])> 0:
                    blanks = []
                    for j in range(len(filtered)):
                        if not (j in self.firing_table[i]):
                            blanks.append(j)
                    self.firing_table_section_ax[0].plot((np.asarray(blanks) * 1000) / fs, [i + 1] * len(blanks), '-')


            self.firing_table_section_canvas.draw()
            self.segmentation_section_canvas.draw()




        else:
            QMessageBox.question(self, 'MUAP Detection', 'No waveforms found',
                                               QMessageBox.Ok, QMessageBox.Ok)

    def classification_control_button_action(self):
        self.classification_control_window.show()

    def initMUAPAnalysisUI(self):
        self.muap_an_tab_layout = QVBoxLayout()

        self.segmentation_section_widget = QGroupBox('EMG Signal Segmentation', self)
        self.segmentation_section_widget_layout = QVBoxLayout()
        self.segmentation_section_widget.setLayout(self.segmentation_section_widget_layout)
        self.segmentation_section_plot_widget = QGroupBox('Signal', self)
        self.segmentation_section_plot_widget_layout = QHBoxLayout()
        self.segmentation_section_plot_widget.setLayout(self.segmentation_section_plot_widget_layout)
        self.segmentation_section_button_widget = QGroupBox('Control Tab', self)
        self.segmentation_section_button_widget_layout = QHBoxLayout()
        self.segmentation_section_button_widget.setMaximumHeight(50)

        self.segmentation_section_button_widget.setLayout(self.segmentation_section_button_widget_layout)
        self.segmentation_section_controls_button = QPushButton('Controls', self)
        self.segmentation_section_controls_button.clicked.connect(self.segmentation_control_button_action)
        self.segmentation_section_detect_peak_button = QPushButton('Detect Peaks', self)
        self.segmentation_section_detect_peak_button.clicked.connect(self.segmentation_detect_peak_button_action)
        self.segmentation_section_validate_peak_button = QPushButton('Validate Peak with Amplitude Change', self)
        self.segmentation_section_validate_peak_button.clicked.connect(self.segmentation_validate_peak_button_action)

        self.segmentation_section_plot_widget_layout.addWidget(self.segmentation_section_canvas)
        self.segmentation_section_button_widget_layout.addWidget(self.segmentation_section_toolbar)
        self.segmentation_section_button_widget_layout.addWidget(self.segmentation_section_controls_button)
        self.segmentation_section_button_widget_layout.addWidget(self.segmentation_section_detect_peak_button)
        self.segmentation_section_button_widget_layout.addWidget(self.segmentation_section_validate_peak_button)
        self.segmentation_section_widget_layout.addWidget(self.segmentation_section_plot_widget)
        self.segmentation_section_widget_layout.addWidget(self.segmentation_section_button_widget)
        self.muap_an_tab_layout.addWidget(self.segmentation_section_widget)

        self.classification_section_widget = QGroupBox('MUAP Classification', self)
        self.classification_section_widget_layout = QVBoxLayout()
        self.classification_section_widget.setLayout(self.classification_section_widget_layout)

        self.classification_section_plot_widget = QGroupBox('MUAP', self)
        self.classification_section_plot_widget_layout = QHBoxLayout()
        self.classification_section_plot_widget.setLayout(self.classification_section_plot_widget_layout)


        #self.classification_section_firing_table_plot_widget = QGroupBox('Firing Table', self)
        #self.classification_section_firing_table_plot_widget_layout = QHBoxLayout()
        #self.classification_section_firing_table_plot_widget.setLayout(self.classification_section_firing_table_plot_widget_layout)
        #self.classification_section_firing_table_plot_widget_layout.addWidget(self.firing_table_section_canvas)

        self.classification_section_plot_classes_scrollarea = QScrollArea(self.parent)
        self.classification_section_plot_classes_widget = QGroupBox('MUAP Classes-Actual(A) and Superimposed(S)', self.classification_section_plot_classes_scrollarea)
        self.classification_section_plot_classes_widget.setMinimumHeight(400)
        self.classification_section_plot_classes_widget.setMinimumWidth(600)
        self.classification_section_plot_classes_scrollarea.setWidget(self.classification_section_plot_classes_widget)
        self.classification_section_plot_classes_scrollarea.setWidgetResizable(True)


        self.classification_section_plot_classes_widget_layout = QVBoxLayout()
        self.classification_section_plot_classes_widget.setLayout(self.classification_section_plot_classes_widget_layout)
        self.classification_section_plot_classes_widget_layout.addWidget(self.classification_section_canvas)
        self.classification_section_plot_widget_layout.addWidget(self.classification_section_plot_classes_scrollarea)
        self.classification_section_plot_widget_layout.addWidget(self.firing_table_section_canvas)

        self.classification_section_toolbar_widget = QGroupBox('Control Tab', self)
        self.classification_section_toolbar_widget.setMaximumHeight(50)
        self.classification_section_toolbar_widget_layout = QHBoxLayout()
        self.classification_section_toolbar_widget.setLayout(self.classification_section_toolbar_widget_layout)
        self.classification_section_control_button = QPushButton('Control', self)
        self.classification_section_control_button.clicked.connect(self.classification_control_button_action)
        self.classification_section_sofm_classification_button = QPushButton('Classify MUAP', self)
        self.classification_section_sofm_classification_button.clicked.connect(self.classification_button_action)
        self.classification_section_toolbar_widget_layout.addWidget(self.classification_section_toolbar)
        self.classification_section_toolbar_widget_layout.addWidget(self.classification_section_control_button)
        self.classification_section_toolbar_widget_layout.addWidget(self.classification_section_sofm_classification_button)
        self.classification_section_toolbar_widget_layout.addWidget(self.firing_table_section_toolbar)

        self.classification_section_widget_layout.addWidget(self.classification_section_plot_widget)
        self.classification_section_widget_layout.addWidget(self.classification_section_toolbar_widget)
        #self.classification_section_widget_layout.addWidget(QLabel("Hello", self))
        self.muap_an_tab_layout.addWidget(self.classification_section_widget)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())