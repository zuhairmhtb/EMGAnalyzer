import sys, math, pdb, collections, _collections, random
from abc import ABC, abstractmethod
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


class ClassificationHandlerThread(QtCore.QThread):
    def __init__(self, dataset_dir, update_training_view_signal, parent):
        QtCore.QThread.__init__(self)
        self.debug_mode = True
        self.data_base_dir = dataset_dir
        self.dataset, self.labels = self.get_dataset()
        c = list(zip(self.dataset, self.labels))
        random.shuffle(c)
        self.dataset, self.labels = zip(*c)
        self.current_data = []
        self.current_labels = []
        self.min_train_data = 10
        self.main_thread = parent
        self.classification_mode = {'Stop': 0, 'Start': 1, 'Paused': 2}
        self.current_classification_mode = self.classification_mode['Stop']
        self.update_training_view_signal = update_training_view_signal
    def get_dataset(self):
        data_type = {}
        total = 0
        urls = []
        labels = []
        for dt in os.listdir(self.data_base_dir):
            dt_p = os.path.join(self.data_base_dir, dt)
            if os.path.isdir(dt_p):
                data_type[dt] = {}
                for disease in os.listdir(dt_p):
                    if "als" in disease.lower():
                        label = 1
                    else:
                        label = 0
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
                                        urls.append(rec_p)
                                        labels.append(label)
                                        total += 1

        self.debug_output(data_type)
        return urls, labels
    def debug_output(self, text):
        if self.debug_mode:
            print(str(text))
    def update_firing_table(self, firing_time, output_class, output_superimposition, firing_table, append=True):
        if not append:
            for i in range(len(output_class)):
                if output_superimposition[i] == 0:
                    firing_table[output_class[i]]= []
        for i in range(len(firing_time)):
            if output_superimposition[i] == 0:
                firing_table[output_class[i]] += firing_time[i]

        self.debug_output(firing_table)
        return firing_table
    def __del__(self):
        self.wait()
    def run(self):
        if self.current_classification_mode == self.classification_mode['Stop']:
            self.current_classification_mode = self.classification_mode['Start']
            current_index = 0
            self.current_data = []
            self.current_labels = []

            current_features = []
            current_muap_features = []
            for ftype in self.main_thread.feature_type:
                for i in range(len(self.main_thread.feature_type[ftype]["features"])):
                    if self.main_thread.feature_type[ftype]["classification_checkboxes"][i].isChecked():
                        current_features.append([ftype, i])

            for i in range(len(self.main_thread.muap_analysis_features)):
                if self.main_thread.muap_analysis_feature_widget_child_widgets[i][4].isChecked():
                    current_muap_features.append(['MUAP', i])
            perform_sofm = self.main_thread.classification_control_window.perform_sofm_learning_checkbox.isChecked()
            perform_lvq = self.main_thread.classification_control_window.perform_lvq_learning_checkbox.isChecked()
            perform_muap_averaging = self.main_thread.classification_control_window.perform_muap_class_averaging_checkbox.isChecked()
            perform_superimposed_decomposition = self.main_thread.classification_control_window.perform_muap_decomposition_checkbox.isChecked()

            output_console_text = ""
            output_console_text += "Current Signal Features: " + str(len(current_features)) + "\n"
            output_console_text += "Current MUAP Features: " + str(len(current_muap_features)) + "\n"
            #self.main_thread.classification_tab_output_label.setText(output_console_text)
            print(output_console_text)

            classification_type = self.main_thread.classification_control_classification_type_select.currentText().lower()

            while (not (self.current_classification_mode == self.classification_mode['Stop'])) and current_index < len(self.dataset):
                while self.current_classification_mode == self.classification_mode['Paused']:
                    self.sleep(2)

                # Load and process data
                d_path = self.dataset[current_index]
                d = np.load(os.path.join(self.dataset[current_index], 'data.npy'))
                fs = self.main_thread.read_sampling_rate(os.path.join(d_path, 'data.hea'))
                cropped_data = self.main_thread.segmentation_control_window.crop_input_data(d)
                filtered = self.main_thread.segmentation_control_window.filter_data(cropped_data, fs)

                signal_feature_data = []
                for i in range(len(current_features)):
                    val = self.main_thread.feature_type[current_features[i][0]]["functions"][current_features[i][1]](
                        filtered, fs, current_features[i][0], current_features[i][1], show_in_tab=False
                    )
                    if not (val is None):
                        signal_feature_data.append(val)
                    else:
                        signal_feature_data.append(sys.maxsize)

                peaks = self.main_thread.segmentation_control_window.detect_peaks(filtered, fs)
                validated_peaks = self.main_thread.segmentation_control_window.validate_peaks(filtered, fs, peaks)
                waveforms, firing_time = self.main_thread.segmentation_control_window.get_muap_waveforms(
                    validated_peaks, filtered, fs
                )

                muap_features = []
                if len(waveforms) > 0:
                    # EMG Decomposition
                    weights = self.main_thread.classification_control_window.get_ann_weights(len(waveforms[0]),
                                                                                 self.main_thread.output_motor_classes, waveforms)
                    if not perform_sofm:
                        weights = self.main_thread.classification_control_window.sofm_learning_phase(waveforms, len(waveforms[0]),
                                                                                         self.main_thread.output_motor_classes,
                                                                                         weights)
                    if not perform_lvq:
                        weights = self.main_thread.classification_control_window.lvq_learning_phase(waveforms, len(waveforms[0]),
                                                                                        self.main_thread.output_motor_classes,
                                                                                        weights)
                    waveform_classes, waveform_superimposition = self.main_thread.classification_control_window.sofm_classification(
                        waveforms, len(waveforms[0]), self.main_thread.output_motor_classes, weights
                    )
                    if not perform_muap_averaging:
                        waveforms, waveform_classes, waveform_superimposition, firing_time = self.main_thread.classification_control_window.average_muap_templates(
                            waveforms, waveform_classes, waveform_superimposition, weights, len(waveforms[0]),
                            self.main_thread.output_motor_classes,
                            firing_time=firing_time
                        )

                    firing_table = [[] for _ in range(self.main_thread.output_motor_classes)]
                    firing_table = self.update_firing_table(firing_time, waveform_classes, waveform_superimposition, firing_table, append=True)
                    if perform_superimposed_decomposition:
                        updated_muaps, residue_superimposed = self.main_thread.classification_control_window.perform_emg_decomposition(
                            waveforms, waveform_classes, waveform_superimposition, firing_time
                        )
                        ft = []
                        classes = []
                        superimposition = [0] * len(updated_muaps)
                        for i in range(len(updated_muaps)):
                            ft.append(updated_muaps[i][2])
                            classes.append(updated_muaps[i][1])
                        firing_table = self.update_firing_table(ft, classes, superimposition, firing_table, append=True)

                    for i in range(len(current_muap_features)):
                        features = self.main_thread.muap_analysis_feature_functions[current_muap_features[i][1]](waveforms, waveform_classes,
                                                                    waveform_superimposition, firing_time, fs, filtered)

                        for j in range(len(features)):
                            if type(features[j]) == list or type(features[j]) == np.ndarray:
                                if len(features[j]) > 0:
                                    features[j] = np.average(features[j])
                                else:
                                    features[j] = 0
                            muap_features.append(features[j])


                else:
                    # Add 0 value for all MUAP Features and Each Motor Unit
                    for i in range(len(current_muap_features)):
                        for _ in range(self.main_thread.output_motor_classes):
                            muap_features.append(0)

                print('Signal Features: ' + str(len(signal_feature_data)))
                print(signal_feature_data)
                print('MUAP Features: ' + str(len(muap_features)))
                print(muap_features)
                if classification_type == "muap classification":
                    self.current_data.append(muap_features)
                elif classification_type == "signal classification":
                    self.current_data.append(signal_feature_data)
                else:
                    self.current_data.append(list(muap_features) + list(signal_feature_data))
                self.current_labels.append(self.labels[current_index])
                current_index += 1


                if len(self.current_data) > self.min_train_data and (not self.main_thread.classifier_view_updated):
                    self.update_training_view_signal.emit()
                    self.sleep(2)


            self.current_classification_mode = self.classification_mode['Stop']
        else:
            self.debug_output('An Instance of the Classification Thread is already running')