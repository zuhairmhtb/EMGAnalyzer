import sys, math, pdb, collections, _collections
from minisom import MiniSom
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMenu, QSizePolicy, QMessageBox,  QCheckBox
from PyQt5.QtWidgets import  QDialog,\
    QHBoxLayout, QLabel, QGroupBox, QSlider, \
    QLineEdit, QScrollArea, QComboBox, QProgressBar
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
import MyNet.temp.emg_editor_pyqt.muap_analysis_functions as analysis_functions
import MyNet.temp.emg_editor_pyqt.signal_analysis_functions as signal_analysis_functions
from MyNet.temp.emg_editor_pyqt.network_widgets import ClassifierWidget, KNearestClassifier, SVMCLassifier, RForestCLassifier
from MyNet.temp.emg_editor_pyqt.classification_handler import ClassificationHandlerThread

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
    update_training_view_signal = QtCore.pyqtSignal()
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
        self.classification_handler = None
        self.classified_accuracies = []
        self.classified_loss = []
        self.classifier_view_updated = False
        self.debug_mode = False
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
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "PreProcessing Tab")
        self.tabs.addTab(self.tab2, "MUAP Analysis Tab")
        self.tabs.addTab(self.tab3, "MUAP Analysis Output Tab")
        self.tabs.addTab(self.tab4, "Signal Analysis Output Tab")
        self.tabs.addTab(self.tab5, "EMG Classification Tab")


        # Create first tab
        self.initPreProcessingUI()
        self.tab1.setLayout(self.preprocessing_tab_layout)

        # Create Second tab
        self.initMUAPAnalysisUI()
        self.tab2.setLayout(self.muap_an_tab_layout)

        # Create Third tab
        self.initMUAPAnalysisOutputUI()
        self.tab3.setLayout(self.muap_analysis_output_tab_layout)

        # Create fourth tab
        self.initSignalAnalysisOutputUI()
        self.tab4.setLayout(self.signal_analysis_output_tab_layout)

        # Create Fifth tab
        self.initClassificationUI()
        self.tab5.setLayout(self.classification_tab_layout)



        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def debug_output(self, text):
        if self.debug_mode:
            print(text)
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

        self.debug_output(self.firing_table)

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

        self.debug_output(data_type)
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
            self.debug_output('Peak Minimum Threshold: ' + str(thresh_per) + '% greater than mean amplitude')
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

        # Update Signal Analysis Output Tab
        for ftype in self.feature_type:
            for i in range(len(self.feature_type[ftype]["features"])):
                if self.feature_type[ftype]["checkboxes"][i].isChecked():
                    self.feature_type[ftype]["functions"][i](filtered, fs, ftype, i)
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
        # Update Signal Analysis Output Tab
        for ftype in self.feature_type:
            for i in range(len(self.feature_type[ftype]["features"])):
                if self.feature_type[ftype]["checkboxes"][i].isChecked():
                    self.feature_type[ftype]["functions"][i](filtered, fs, ftype, i)

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

        # Update Signal Analysis Output Tab
        for ftype in self.feature_type:
            for i in range(len(self.feature_type[ftype]["features"])):
                if self.feature_type[ftype]["checkboxes"][i].isChecked():
                    self.feature_type[ftype]["functions"][i](filtered, fs, ftype, i)

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

            # Update Analysis Output Tab
            print('Analyzing MUAP...')
            for i in range(len(self.muap_analysis_features)):

                if self.muap_analysis_feature_widget_child_widgets[i][3].isChecked():
                    self.muap_analysis_feature_functions[i](waveforms, waveform_classes, waveform_superimposition, firing_time, fs, filtered)
            print('MUAP Analysis complete...')
            # Update Analysis Tab
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
                    self.debug_output('Motor unit: ' + str(i+1))
                    self.debug_output('Firing time: ' + str(self.firing_table[i]))
                    self.debug_output('Total fires: ' + str(len(self.firing_table[i])))
                    self.debug_output('Total data points: ' + str(len(filtered)))

                    self.firing_table_section_ax[0].plot( (np.asarray(self.firing_table[i])*1000)/fs, [i+1]*len(self.firing_table[i]), 'x')
                    self.segmentation_section_ax[0].plot(self.firing_table[i], np.asarray(filtered)[self.firing_table[i]], 'o')

            """for i in range(len(self.firing_table)):
                if len(self.firing_table[i])> 0:
                    blanks = []
                    for j in range(len(filtered)):
                        if not (j in self.firing_table[i]):
                            blanks.append(j)
                    self.firing_table_section_ax[0].plot((np.asarray(blanks) * 1000) / fs, [i + 1] * len(blanks), '-')"""


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

    def calculate_amp_difference(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        peaks_thresh = -1
        window = self.segmentation_control_window.segment_window
        muaps = np.asarray(muaps)
        output_classes = np.asarray(output_classes)
        output_superimposition = np.asarray(output_superimposition)

        actual_muaps = muaps[output_superimposition == 0]
        actual_output_classes = output_classes[output_superimposition == 0]

        amp_difference = analysis_functions.calculate_amplitude_difference(actual_muaps, peaks_thresh)
        outputs = ["" for _ in range(self.output_motor_classes)]
        outputs_val = [ [] for _ in range(self.output_motor_classes)]
        for i in range(len(actual_muaps)):
            outputs[actual_output_classes[i]] = outputs[actual_output_classes[i]] + "{0:.5f}".format(amp_difference[i]) + "\n"
            outputs_val[actual_output_classes[i]].append(amp_difference[i])
        if show_in_tab:
            for i in range(len(outputs)):
                self.muap_analysis_feature_widget_child_widgets[0][2][i].setText(outputs[i])
                #if len(outputs_val[i]) > 0:
                    #outputs_val[i] = np.average(outputs_val)
        return outputs_val

    def calculate_duration(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        peaks_thresh = -1
        window = self.segmentation_control_window.segment_window
        muaps = np.asarray(muaps)
        output_classes = np.asarray(output_classes)
        output_superimposition = np.asarray(output_superimposition)

        actual_muaps = muaps[output_superimposition == 0]
        actual_output_classes = output_classes[output_superimposition == 0]

        durations, end_points = analysis_functions.calculate_waveform_duration(actual_muaps, window)
        outputs = ["" for _ in range(self.output_motor_classes)]
        outputs_val = [[] for _ in range(self.output_motor_classes)]
        for i in range(len(actual_muaps)):
            outputs[actual_output_classes[i]] = outputs[actual_output_classes[i]] + "{0:.5f}".format(durations[i]) + "\n"
            outputs_val[actual_output_classes[i]].append(durations[i])
        if show_in_tab:
            for i in range(len(outputs)):
                self.muap_analysis_feature_widget_child_widgets[1][2][i].setText(outputs[i])
                #if len(outputs_val[i]) > 0:
                    #outputs_val[i] = np.average(outputs_val)
        return outputs_val
    def calculate_rect_area(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        peaks_thresh = -1
        window = self.segmentation_control_window.segment_window
        muaps = np.asarray(muaps)
        output_classes = np.asarray(output_classes)
        output_superimposition = np.asarray(output_superimposition)

        actual_muaps = muaps[output_superimposition == 0]
        actual_output_classes = output_classes[output_superimposition == 0]

        rect_area = analysis_functions.calculate_rectified_waveform_area(actual_muaps, window)
        outputs = ["" for _ in range(self.output_motor_classes)]
        outputs_val = [[] for _ in range(self.output_motor_classes)]
        for i in range(len(actual_muaps)):
            outputs[actual_output_classes[i]] = outputs[actual_output_classes[i]] + "{0:.5f}".format(rect_area[i]) + "\n"
            outputs_val[actual_output_classes[i]].append(rect_area[i])
        if show_in_tab:
            for i in range(len(outputs)):
                self.muap_analysis_feature_widget_child_widgets[2][2][i].setText(outputs[i])
                #if len(outputs_val[i]) > 0:
                    #outputs_val[i] = np.average(outputs_val)
        return outputs_val
    def calculate_rise_time(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        peaks_thresh = -1
        window = self.segmentation_control_window.segment_window
        muaps = np.asarray(muaps)
        output_classes = np.asarray(output_classes)
        output_superimposition = np.asarray(output_superimposition)

        actual_muaps = muaps[output_superimposition == 0]
        actual_output_classes = output_classes[output_superimposition == 0]

        rise_time = analysis_functions.calculate_rise_time(actual_muaps, window, peaks_thresh)
        outputs = ["" for _ in range(self.output_motor_classes)]
        outputs_val = [[] for _ in range(self.output_motor_classes)]
        for i in range(len(actual_muaps)):
            outputs[actual_output_classes[i]] = outputs[actual_output_classes[i]] + "{0:.5f}".format(rise_time[i]) + "\n"
            outputs_val[actual_output_classes[i]].append(rise_time[i])
        if show_in_tab:
            for i in range(len(outputs)):
                self.muap_analysis_feature_widget_child_widgets[3][2][i].setText(outputs[i])
                #if len(outputs_val[i]) > 0:
                    #outputs_val[i] = np.average(outputs_val)
        return outputs_val

    def calculate_phases(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        peaks_thresh = -1
        window = self.segmentation_control_window.segment_window
        muaps = np.asarray(muaps)
        output_classes = np.asarray(output_classes)
        output_superimposition = np.asarray(output_superimposition)

        actual_muaps = muaps[output_superimposition == 0]
        actual_output_classes = output_classes[output_superimposition == 0]

        phases = analysis_functions.calculate_phase(actual_muaps, window)
        outputs = ["" for _ in range(self.output_motor_classes)]
        outputs_val = [[] for _ in range(self.output_motor_classes)]
        for i in range(len(actual_muaps)):
            outputs[actual_output_classes[i]] = outputs[actual_output_classes[i]] + "{0:.5f}".format(phases[i]) + "\n"
            outputs_val[actual_output_classes[i]].append(phases[i])
        if show_in_tab:
            for i in range(len(outputs)):
                self.muap_analysis_feature_widget_child_widgets[4][2][i].setText(outputs[i])
                #if len(outputs_val[i]) > 0:
                    #outputs_val[i] = np.average(outputs_val)
        return outputs_val

    def calculate_turns(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        peaks_thresh = -1
        window = self.segmentation_control_window.segment_window
        muaps = np.asarray(muaps)
        output_classes = np.asarray(output_classes)
        output_superimposition = np.asarray(output_superimposition)

        actual_muaps = muaps[output_superimposition == 0]
        actual_output_classes = output_classes[output_superimposition == 0]

        turns = analysis_functions.calculate_turns(actual_muaps, window, peaks_thresh)
        outputs = ["" for _ in range(self.output_motor_classes)]
        outputs_val = [[] for _ in range(self.output_motor_classes)]
        for i in range(len(actual_muaps)):
            outputs[actual_output_classes[i]] = outputs[actual_output_classes[i]] + "{0:.5f}".format(turns[i]) + "\n"
            outputs_val[actual_output_classes[i]].append(turns[i])
        if show_in_tab:
            for i in range(len(outputs)):
                self.muap_analysis_feature_widget_child_widgets[5][2][i].setText(outputs[i])
                #if len(outputs_val[i]) > 0:
                 #   outputs_val[i] = np.average(outputs_val)
        return outputs_val

    def calculate_firing_rate(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        total_duration = len(signal) / fs
        firing_rates = []
        for i in range(self.output_motor_classes):
            firing_pattern = self.firing_table[i]
            total = len(firing_pattern)
            firing_rate = total / total_duration
            firing_rates.append(firing_rate)
            if show_in_tab:
                self.muap_analysis_feature_widget_child_widgets[6][2][i].setText("{0:.5f}".format(firing_rate))
        return firing_rates
    def calculate_interspike_interval(self):
        interspike_intervals = []
        for i in range(self.output_motor_classes):
            firing_pattern = self.firing_table[i]
            interspike_intervals.append([self.firing_table[i][j+1] - self.firing_table[i][j] for j in range(len(self.firing_table[i])-1)])
        return interspike_intervals

    def calculate_mad_isi(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):

        isi = self.calculate_interspike_interval()

        mads = []
        for i in range(self.output_motor_classes):
            if len(isi[i]) > 0:
                intervals = np.asarray(isi[i])
                intervals = (intervals*1000)/fs
                mean_isi = np.mean(intervals)
                deviation = np.abs(intervals - mean_isi)
                mad = np.sum(deviation)/len(deviation)
            else:
                mad = 0
            mads.append(mad)
            if show_in_tab:
                self.muap_analysis_feature_widget_child_widgets[7][2][i].setText("{0:.5f}".format(mad))
        return mads

    def calculate_std_isi(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        isi = self.calculate_interspike_interval()

        stds = []
        for i in range(self.output_motor_classes):
            if len(isi[i]) > 0:
                intervals = np.asarray(isi[i])
                intervals = (intervals * 1000) / fs
                std = np.std(intervals)
            else:
                std = 0
            stds.append(std)
            if show_in_tab:
                self.muap_analysis_feature_widget_child_widgets[8][2][i].setText("{0:.5f}".format(std))
        return stds

    def calculate_mean_isi(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        isi = self.calculate_interspike_interval()

        means = []
        for i in range(self.output_motor_classes):
            if len(isi[i]) > 0:
                intervals = np.asarray(isi[i])
                intervals = (intervals * 1000) / fs
                mean = np.mean(intervals)

            else:
                mean = 0
            means.append(mean)
            if show_in_tab:
                self.muap_analysis_feature_widget_child_widgets[9][2][i].setText("{0:.5f}".format(mean))
        return means

    def calculate_freq_tlocked_muap(self, muaps, output_classes, output_superimposition, firing_time, fs, signal, show_in_tab=True):
        freqs = []
        for i in range(self.output_motor_classes):
            firing_pattern = self.firing_table[i]
            if(len(firing_pattern) > 0):
                total = 0
                for j in range(len(firing_pattern)):
                    for k in range(len(self.firing_table)):
                        if k != i and j in self.firing_table[k]:
                            total += 1
                freq = total/(len(signal) / fs)
            else:
                freq = 0
            freqs.append(freq)
            if show_in_tab:
                self.muap_analysis_feature_widget_child_widgets[10][2][i].setText("{0:.5f}".format(freq))
        return freqs

    def initMUAPAnalysisOutputUI(self):
        self.muap_analysis_output_tab_layout = QHBoxLayout()

        self.muap_waveform_analysis_widget = QGroupBox('MUAP Waveform Analysis')
        self.muap_waveform_analysis_widget_layout = QVBoxLayout()
        self.muap_waveform_analysis_widget.setLayout(self.muap_waveform_analysis_widget_layout)
        self.firing_table_analysis_widget = QGroupBox('Firing Table Analysis')
        self.firing_table_analysis_widget_layout = QVBoxLayout()
        self.firing_table_analysis_widget.setLayout(self.firing_table_analysis_widget_layout)

        self.muap_analysis_output_tab_layout.addWidget(self.muap_waveform_analysis_widget)
        self.muap_analysis_output_tab_layout.addWidget(self.firing_table_analysis_widget)

        self.muap_analysis_features = ["Amplitude Difference(uV):Min -ve and Max+ve Peaks",
                                       "Duration(ms)",
                                       "Rectified Area(Integrated over calculated duration)",
                                       "Rise Time(ms):Time difference between Max -ve and preceeding Min +ve Peak",
                                       "Phases",
                                       "Turns(No. of +ve and -ve Peaks)",
                                       "Firing Rate",
                                       "Mean Absolute Deviation of Inter Spike Interval(ISI)(ms)",
                                       "Standard Deviation of ISI",
                                       "Mean of ISI",
                                       "Frequency of Time locked MUAPs(seconds)",
                                       ]
        self.muap_analysis_feature_type = [
            "waveform", "waveform", "waveform", "waveform", "waveform", "waveform",
            "firing_table", "firing_table", "firing_table", "firing_table", "firing_table"
        ]
        self.muap_analysis_feature_functions = [self.calculate_amp_difference,
                                                self.calculate_duration,
                                                self.calculate_rect_area,
                                                self.calculate_rise_time,
                                                self.calculate_phases,
                                                self.calculate_turns,
                                                self.calculate_firing_rate,
                                                self.calculate_mad_isi,
                                                self.calculate_std_isi,
                                                self.calculate_mean_isi,
                                                self.calculate_freq_tlocked_muap]

        self.muap_analysis_feature_scrollareas = [QScrollArea(self.parent) for _ in range(len(self.muap_analysis_features))]
        self.muap_analysis_feature_widgets = [QGroupBox("MUAP Waveform " + self.muap_analysis_features[i], self.muap_analysis_feature_scrollareas[i])
                                              for i in range(len(self.muap_analysis_features))]
        self.muap_analysis_feature_widget_layouts = [QHBoxLayout() for _ in range(len(self.muap_analysis_features))]

        self.muap_analysis_feature_widget_child_widgets = [
            [
                [
                    QGroupBox('MU ' + str(i), self) for i in range(self.output_motor_classes)
                ],
                [
                    QVBoxLayout() for _ in range(self.output_motor_classes)
                ],
                [
                    QLabel("N/A", self) for _ in range(self.output_motor_classes)
                ],
                QCheckBox("Calculate Parameters", self),
                QCheckBox("Classifier", self)

            ] for _ in range(len(self.muap_analysis_features))
        ]
        for i in range(len(self.muap_analysis_features)):

            self.muap_analysis_feature_scrollareas[i].setWidgetResizable(True)
            self.muap_analysis_feature_scrollareas[i].setWidget(self.muap_analysis_feature_widgets[i])
            self.muap_analysis_feature_widgets[i].setLayout(self.muap_analysis_feature_widget_layouts[i])
            self.muap_analysis_feature_widget_child_widgets[i][3].setChecked(True)
            self.muap_analysis_feature_widget_child_widgets[i][4].setChecked(True)
            for j in range(self.output_motor_classes):
                self.muap_analysis_feature_widget_child_widgets[i][2][j].setWordWrap(True)
                self.muap_analysis_feature_widget_child_widgets[i][0][j].setLayout(self.muap_analysis_feature_widget_child_widgets[i][1][j])
                self.muap_analysis_feature_widget_child_widgets[i][1][j].addWidget(self.muap_analysis_feature_widget_child_widgets[i][2][j])
                self.muap_analysis_feature_widget_layouts[i].addWidget(self.muap_analysis_feature_widget_child_widgets[i][0][j])
            self.muap_analysis_feature_widget_layouts[i].addWidget(self.muap_analysis_feature_widget_child_widgets[i][3])
            self.muap_analysis_feature_widget_layouts[i].addWidget(
                self.muap_analysis_feature_widget_child_widgets[i][4])
            if self.muap_analysis_feature_type[i] == "waveform":
                self.muap_waveform_analysis_widget_layout.addWidget(self.muap_analysis_feature_scrollareas[i])
            elif self.muap_analysis_feature_type[i] == "firing_table":
                self.firing_table_analysis_widget_layout.addWidget(self.muap_analysis_feature_scrollareas[i])

    def calculate_mad(self, signal, fs, ftype, index, show_in_tab=True):
        mad = signal_analysis_functions.calculate_mav(signal)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(mad))
        return mad
    def calculate_rms(self, signal, fs, ftype, index, show_in_tab=True):
        rms = signal_analysis_functions.calculate_rms(signal)
        self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(rms))
    def calculate_sd(self, signal, fs, ftype, index, show_in_tab=True):
        sd = signal_analysis_functions.calculate_std(signal)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(sd))
        return sd
    def calculate_var(self, signal, fs, ftype, index, show_in_tab=True):
        var = np.var(signal)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(var))
        return var
    def calculate_aac(self, signal, fs, ftype, index, show_in_tab=True):
        aac = signal_analysis_functions.calculate_aac(signal)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(aac))
        return aac
    def calculate_smad(self, signal, fs, ftype, index, show_in_tab=True):
        smad = signal_analysis_functions.calculate_smad(signal, fs, self.segmentation_control_window.segment_window)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(smad))
        return smad
    def calculate_energy(self, signal, fs, ftype, index, show_in_tab=True):
        eng = np.sum(np.square(np.abs(signal)))
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(eng))
        return eng

    def calculate_mnf(self, signal, fs, ftype, index, show_in_tab=True):

        sxx = np.abs(np.fft.fft(signal))
        f = np.fft.fftfreq(len(signal), 1/fs)
        mnf = signal_analysis_functions.calculate_mnf(sxx, f)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(mnf))
        return mnf

    def calculate_mdf(self, signal, fs, ftype, index, show_in_tab=True):
        sxx = np.abs(np.fft.fft(signal))
        f = np.fft.fftfreq(len(signal), 1 / fs)
        mdf = signal_analysis_functions.calculate_mdf(sxx, f)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(mdf))
        return mdf

    def calculate_amd(self, signal, fs, ftype, index, show_in_tab=True):

        window = self.segmentation_control_window.segment_window
        window_len = int((fs*window)/1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='density')
        amd = signal_analysis_functions.calculate_amd(sxx, t)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(amd))
        return amd

    def calculate_vmf(self, signal, fs, ftype, index, show_in_tab=True):
        window = self.segmentation_control_window.segment_window
        window_len = int((fs*window)/1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='density')
        vmf = signal_analysis_functions.calculate_vmf(sxx, f, t)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(vmf))
        return vmf

    def calculate_mnp(self, signal, fs, ftype, index, show_in_tab=True):
        window = self.segmentation_control_window.segment_window
        window_len = int((fs*window)/1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='spectrum')
        mnp = signal_analysis_functions.calculate_mnp(sxx, t)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(mnp))
        return mnp


    def calculate_pkf(self, signal, fs, ftype, index, show_in_tab=True):
        window = self.segmentation_control_window.segment_window
        window_len = int((fs*window)/1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='spectrum')
        pkf = signal_analysis_functions.calculate_pkf(sxx, f, t)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(pkf))
        return pkf

    def calculate_top(self, signal, fs, ftype, index, show_in_tab=True):
        window = self.segmentation_control_window.segment_window
        window_len = int((fs*window)/1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='spectrum')
        top = np.sum(sxx.flatten())
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(top))
        return top


    def calculate_vcf(self, signal, fs, ftype, index, show_in_tab=True):
        window = self.segmentation_control_window.segment_window
        window_len = int((fs*window)/1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='spectrum')
        vcf = signal_analysis_functions.calculate_vcf(sxx, f, t)
        if show_in_tab:
            self.feature_type[ftype]["labels"][index].setText("{0:.5f}".format(vcf))
        return vcf

    def calculate_tpf(self, signal, fs, ftype, index, show_in_tab=True):
        window = self.segmentation_control_window.segment_window
        window_len = int((fs * window) / 1000)
        f, t, sxx = spectrogram(signal, fs, nperseg=window_len, mode='psd', scaling='density')
        peaks, peaks_img = signal_analysis_functions.calculate_npf(sxx, np.mean(sxx.flatten()), window_len)
        total = 0
        for i in range(len(peaks)):
            total += len(peaks[i])
        return total


    def initSignalAnalysisOutputUI(self):
        self.signal_analysis_output_tab_layout = QVBoxLayout()

        # Time domain Features
        self.feature_type = {
            "Time Domain": {
                "features": ["Mean Absolute Deviation", "Root Mean Square", "Standard Deviation", "Varaince",
                                     "Average Amplitude Change", "Slope of Mean Absolute Value", "Energy"],
                "functions": [self.calculate_mad, self.calculate_rms, self.calculate_sd, self.calculate_var,
                              self.calculate_aac, self.calculate_smad, self.calculate_energy],
                "checkboxes": [],
                "classification_checkboxes": [],
                "labels": [],
                "widgets": [],
                "layouts": [],
                "parent_widget": None,
                "parent_layout": QHBoxLayout()
            },
            "Feature Domain": {
                "features": ["Mean Frequency", "Median Frequency", "Average Maximum Density",
                             "Variance of Maximum Frequency", "Mean Power", "Peak Frequency",
                             "Total Power", "Variance of Central Frequency", "Number of Peaks"
                             ],
                "functions": [self.calculate_mnf, self.calculate_mdf, self.calculate_amd,
                              self.calculate_vmf, self.calculate_mnp, self.calculate_pkf,
                              self.calculate_top, self.calculate_vcf, self.calculate_tpf],
                "checkboxes": [],
                "classification_checkboxes": [],
                "labels": [],
                "widgets": [],
                "layouts": [],
                "parent_widget": None,
                "parent_layout": QHBoxLayout()
            }
        }

        # Create View for each feature type
        for ftype in self.feature_type:
            parent_widget = QGroupBox(ftype.upper() + " ANALYSIS", self)
            parent_widget.setLayout(self.feature_type[ftype]["parent_layout"])

            for j in range(len(self.feature_type[ftype]["features"])):
                widget = QGroupBox(self.feature_type[ftype]["features"][j], self)
                layout = QHBoxLayout()
                widget.setLayout(layout)
                checkbox = QCheckBox("Calculate", self)
                checkbox.setChecked(True)
                classification_checkbox = QCheckBox('Classifier', self)
                classification_checkbox.setChecked(True)
                label = QLabel("N/A", self)
                layout.addWidget(checkbox)
                layout.addWidget(label)
                layout.addWidget(classification_checkbox)
                self.feature_type[ftype]["parent_layout"].addWidget(widget)
                self.feature_type[ftype]["widgets"].append(widget)
                self.feature_type[ftype]["layouts"].append(layout)
                self.feature_type[ftype]["checkboxes"].append(checkbox)
                self.feature_type[ftype]["labels"].append(label)
                self.feature_type[ftype]["classification_checkboxes"].append(classification_checkbox)

            self.signal_analysis_output_tab_layout.addWidget(parent_widget)
            self.feature_type[ftype]["parent_widget"] = parent_widget


    def start_emg_classification_thread(self):
        for k in self.classifiers:
            self.classifiers[k].view.graphs['accuracy']['axis'].clear()
            self.classifiers[k].view.graphs['average accuracy']['axis'].clear()
            self.classifiers[k].view.graphs['data']['axis'].clear()
        if not (self.classification_handler is None):
            if self.classification_handler.current_classification_mode == self.classification_handler.classification_mode['Paused']:
                self.classification_handler.current_classification_mode = self.classification_handler.classification_mode['Start']

            elif self.classification_handler.current_classification_mode == self.classification_handler.classification_mode['Stop']:
                self.classification_handler = ClassificationHandlerThread(str(self.classification_control_dataset_url.text()),
                                                                          self.update_training_view_signal, self)
                self.classification_handler.update_training_view_signal.connect(self.update_training_view)
                self.classified_accuracies = [ [] for _ in range(len(self.classifiers))]
                self.classified_loss = [[] for _ in range(len(self.classifiers))]
                self.classifier_view_updated = False
                for clf in self.classifiers:
                    self.classifiers[clf].create_classifier()
                self.classification_handler.start()
        else:
            self.classification_handler = ClassificationHandlerThread(str(self.classification_control_dataset_url.text()),
                                                                      self.update_training_view_signal, self)
            self.classification_handler.update_training_view_signal.connect(self.update_training_view)
            self.classified_accuracies = [[] for _ in range(len(self.classifiers))]
            self.classified_loss = [[] for _ in range(len(self.classifiers))]
            self.classifier_view_updated = False
            for clf in self.classifiers:
                self.classifiers[clf].create_classifier()
            self.classification_handler.start()

    def pause_emg_classification_thread(self):
        if not (self.classification_handler is None) and self.classification_handler.current_classification_mode == self.classification_handler.classification_mode['Start']:
            self.classification_handler.current_classification_mode = self.classification_handler.classification_mode['Paused']

    def stop_emg_classification_thread(self):
        if not (self.classification_handler is None):
            if self.classification_handler.current_classification_mode != self.classification_handler.classification_mode['Stop']:
                self.classification_handler.current_classification_mode = self.classification_handler.classification_mode['Stop']


    def update_training_view(self):
        features = self.classification_handler.current_data
        features_np = np.asarray(features)
        labels = self.classification_handler.current_labels
        labels_np = np.asarray(labels)
        if(len(features) > self.classification_handler.min_train_data):
            self.classifier_view_updated = True
            accuracies = []
            losses = []
            c = 0
            for k in self.classifiers:
                print('Training with features of shape' + str(features_np.shape))
                print(features)
                accuracy, loss = self.classifiers[k].train_classifier(features, labels)
                print('Classifier ' + k.upper() + ":")
                print("Accuracy: " + str(accuracy))

                self.classified_accuracies[c].append(accuracy)
                self.classified_loss[c].append(loss)

                print('Average Accuracy: ' + str(np.average(self.classified_accuracies[c])))
                print('Dataset length: ' + str(len(features)))

                self.classifiers[k].view.info_type["training"]["children"]["accuracy"]["value"].setText("{0:.3f}".format(accuracy*100))
                self.classifiers[k].view.info_type["training"]["children"]["loss"]["value"].setText(
                    "{0:.3f}".format(loss * 100))
                self.classifiers[k].view.graphs['accuracy']['axis'].clear()
                self.classifiers[k].view.graphs['accuracy']["axis"].plot(self.classified_accuracies[c])
                self.classifiers[k].view.graphs['accuracy']["axis"].grid(b=True)
                self.classifiers[k].view.graphs['accuracy']["canvas"].draw()
                #self.classifiers[k].view.graphs['average accuracy']['axis'].clear()
                self.classifiers[k].view.graphs['average accuracy']["axis"].scatter(
                    len(self.classified_accuracies[c]), np.average(self.classified_accuracies[c]))
                self.classifiers[k].view.graphs['average accuracy']["axis"].grid(b=True)
                self.classifiers[k].view.graphs['average accuracy']["canvas"].draw()
                self.classifiers[k].view.graphs['data']['axis'].clear()
                #for i in range(features_np.shape[2]):
                 #   self.classifiers[k].view.graphs['data']['axis'].scatter(features_np[:, 0, i], features_np[:, 1, i],
                  #                                                          c=labels_np + 1+i)
                self.classifiers[k].view.graphs['data']['axis'].scatter(features_np[:, 0], features_np[:, 1],
                                                                        c=labels_np + 1)
                self.classifiers[k].view.graphs['data']['axis'].grid(True)
                self.classifiers[k].view.graphs['data']["canvas"].draw()
                c += 1
            self.classifier_view_updated = False
        if not (self.classification_handler is None):
            self.classification_control_progressbar.setValue(
                int(len(self.classification_handler.current_data)*100/len(self.classification_handler.dataset))
            )


    def initClassificationUI(self):
        self.classification_tab_layout = QVBoxLayout()

        self.classification_tab_control_widget = QGroupBox('Control Tab', self)
        self.classification_tab_control_widget_layout = QHBoxLayout()
        self.classification_tab_control_widget.setLayout(self.classification_tab_control_widget_layout)

        self.classification_control_progressbar = QProgressBar(self)
        self.classification_control_progressbar.setMinimum(0)
        self.classification_control_progressbar.setMaximum(100)
        self.classification_control_progressbar_notification = QLabel("Status: N/A", self)
        self.classification_control_progressbar.setGeometry(200, 80, 250, 20)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_progressbar)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_progressbar_notification)

        self.classification_control_start_btn = QPushButton("Start", self)
        self.classification_control_start_btn.clicked.connect(self.start_emg_classification_thread)
        self.classification_control_pause_btn = QPushButton("Pause", self)
        self.classification_control_pause_btn.clicked.connect(self.pause_emg_classification_thread)
        self.classification_control_stop_btn = QPushButton("Stop", self)
        self.classification_control_stop_btn.clicked.connect(self.stop_emg_classification_thread)
        self.classification_control_classification_type_select = QComboBox(self)
        self.classification_control_classification_type_select.addItem('MUAP Classification')
        self.classification_control_classification_type_select.addItem('Signal Classification')
        self.classification_control_classification_type_select.addItem('Both')

        self.classification_tab_control_widget_layout.addWidget(self.classification_control_start_btn)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_pause_btn)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_stop_btn)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_classification_type_select)

        self.classification_control_dataset_label = QLabel('Dataset URL:', self)
        self.classification_control_dataset_url = QLineEdit(self)
        self.classification_control_dataset_url.setText(self.data_base_dir)
        self.classification_control_dataset_url.setText(self.data_base_dir)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_dataset_label)
        self.classification_tab_control_widget_layout.addWidget(self.classification_control_dataset_url)

        self.classification_tab_network_widget = QGroupBox('Network Tab', self)
        self.classification_tab_network_widget_layout = QVBoxLayout()
        self.classification_tab_network_widget.setLayout(self.classification_tab_network_widget_layout)
        self.classifiers = {
            'K Nearest Neighbor': KNearestClassifier('K Nearest Neigbors', 1, self),
            'Support Vector Machine': SVMCLassifier('Support Vector Machine', 2, self),
            'Random Forest': RForestCLassifier('Random Forest', 3, self)
        }
        self.classified_accuracies = [[] for _ in range(len(self.classifiers))]
        self.classified_loss = [[] for _ in range(len(self.classifiers))]
        widget_group = None
        layout_group = None
        classifier_number = 0
        for k in sorted(self.classifiers):
            classifier_scrollview = self.classifiers[k].view
            if classifier_number % 2 == 0:
                widget_group = QGroupBox("", self)
                layout_group = QHBoxLayout()
                widget_group.setLayout(layout_group)
                self.classification_tab_network_widget_layout.addWidget(widget_group)
            layout_group.addWidget(classifier_scrollview.main_scroller)
            classifier_number += 1




        self.classification_tab_output_scrollarea = QScrollArea(self.parent)
        self.classification_tab_output_scrollarea.setMinimumHeight(100)

        self.classification_tab_output_scrollarea.setWidgetResizable(True)
        self.classification_tab_output_widget = QGroupBox('Output Console', self.classification_tab_output_scrollarea)
        self.classification_tab_output_scrollarea.setWidget(self.classification_tab_output_widget)
        self.classification_tab_output_widget_layout = QHBoxLayout()
        self.classification_tab_output_widget.setLayout(self.classification_tab_output_widget_layout)

        self.classification_tab_output_label = QLabel("...............................")
        self.classification_tab_output_label.setWordWrap(True)


        self.classification_tab_layout.addWidget(self.classification_tab_control_widget)
        self.classification_tab_layout.addWidget(self.classification_tab_network_widget)
        self.classification_tab_layout.addWidget(self.classification_tab_output_scrollarea)







if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())