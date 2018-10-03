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

class ClassificationControlWinow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.ann_init_weight = 0.0001
        self.ann_init_midweight_times = 0.01
        self.learning_rate = 1
        self.lvq_gaussian_hk = 0.2
        self.lvq_gaussina_thresh = 0.005
        self.lvq_second_winner_adapt_thresh = 0.1
        self.lvq_gaussian_hk_adjust_thresh = 0.01
        self.classification_superimposition_thresh = 0.2
        self.classification_muap_averaging_amount = 3


        self.layout = QVBoxLayout()
        self.ann_param_widget = QGroupBox('Artificial Neural Network Parameters', self)
        self.ann_param_widget_layout = QHBoxLayout()
        self.ann_param_widget.setLayout(self.ann_param_widget_layout)
        self.lvq_param_widget = QGroupBox('LVQ2 Parameters', self)
        self.lvq_param_widget_layout = QHBoxLayout()
        self.lvq_param_widget.setLayout(self.lvq_param_widget_layout)
        self.sofm_classification_param_widget = QGroupBox('SOFM Parameters', self)
        self.sofm_classification_param_widget_layout = QHBoxLayout()
        self.sofm_classification_param_widget.setLayout(self.sofm_classification_param_widget_layout)
        self.sofm_classification_param_widget_left = QGroupBox('', self)
        self.sofm_classification_param_widget_left_layout = QVBoxLayout()
        self.sofm_classification_param_widget_left.setLayout(self.sofm_classification_param_widget_left_layout)
        self.sofm_classification_param_widget_right = QGroupBox('', self)
        self.sofm_classification_param_widget_right_layout = QVBoxLayout()
        self.sofm_classification_param_widget_right.setLayout(self.sofm_classification_param_widget_right_layout)
        self.sofm_classification_param_widget_layout.addWidget(self.sofm_classification_param_widget_left)
        self.sofm_classification_param_widget_layout.addWidget(self.sofm_classification_param_widget_right)


        self.ann_init_weight_label = QLabel('Initial Weight: ', self)
        self.ann_init_weight_input = QLineEdit(self)
        self.ann_init_midweight_label = QLabel('Bias for Middle Node: ', self)
        self.ann_init_midweight_input = QLineEdit(self)
        self.ann_param_widget_layout.addWidget(self.ann_init_weight_label)
        self.ann_param_widget_layout.addWidget(self.ann_init_weight_input)
        self.ann_param_widget_layout.addWidget(self.ann_init_midweight_label)
        self.ann_param_widget_layout.addWidget(self.ann_init_midweight_input)

        self.lvq_hk_label = QLabel('Initial Gaussian Value: ', self)
        self.lvq_hk_input = QLineEdit(self)
        self.lvq_hk_thresh_label = QLabel('Gaussian Threshold: ', self)
        self.lvq_hk_thresh_input = QLineEdit(self)
        self.lvq_second_winner_adapt_thresh_label = QLabel('Adaptation (Second Winner Threshold): ', self)
        self.lvq_second_winner_adapt_thresh_input = QLineEdit(self)
        self.lvq_hk_adjust_thresh_label = QLabel('Gaussian Adaptation Threshold: ', self)
        self.lvq_hk_adjust_thresh_input = QLineEdit(self)
        self.lvq_left_widget = QGroupBox('', self)
        self.lvq_left_widget_layout = QVBoxLayout()
        self.lvq_left_widget.setLayout(self.lvq_left_widget_layout)
        self.lvq_right_widget = QGroupBox('', self)
        self.lvq_right_widget_layout = QVBoxLayout()
        self.lvq_right_widget.setLayout(self.lvq_right_widget_layout)
        self.lvq_left_widget_layout.addWidget(self.lvq_hk_label)
        self.lvq_left_widget_layout.addWidget(self.lvq_hk_input)
        self.lvq_left_widget_layout.addWidget(self.lvq_hk_thresh_label)
        self.lvq_left_widget_layout.addWidget(self.lvq_hk_thresh_input)
        self.lvq_right_widget_layout.addWidget(self.lvq_second_winner_adapt_thresh_label)
        self.lvq_right_widget_layout.addWidget(self.lvq_second_winner_adapt_thresh_input)
        self.lvq_right_widget_layout.addWidget(self.lvq_hk_adjust_thresh_label)
        self.lvq_right_widget_layout.addWidget(self.lvq_hk_adjust_thresh_input)
        self.lvq_param_widget_layout.addWidget(self.lvq_left_widget)
        self.lvq_param_widget_layout.addWidget(self.lvq_right_widget)

        self.classification_superimposition_thresh_label = QLabel('Superimposition Calculation Threshold: ', self)
        self.classification_superimposition_thresh_input = QLineEdit(self)
        self.classification_learning_rate_label = QLabel('Learning rate: ', self)
        self.classification_learning_rate_input = QLineEdit(self)
        self.classification_muap_averaging_amount_label = QLabel('Max No. of MUAPs in a class: ', self)
        self.classification_muap_averaging_amount_input = QLineEdit(self)
        self.perform_sofm_learning_checkbox = QCheckBox('Disable SOFM Learning Phase', self)
        self.perform_lvq_learning_checkbox = QCheckBox('Disable LVQ Learning Phase', self)
        self.perform_muap_class_averaging_checkbox = QCheckBox('Disable MUAP Averaging per class', self)
        self.perform_muap_decomposition_checkbox = QCheckBox('EMG Decomposition', self)
        self.sofm_classification_param_widget_left_layout.addWidget(self.classification_superimposition_thresh_label)
        self.sofm_classification_param_widget_left_layout.addWidget(self.classification_superimposition_thresh_input)
        self.sofm_classification_param_widget_left_layout.addWidget(self.classification_learning_rate_label)
        self.sofm_classification_param_widget_left_layout.addWidget(self.classification_learning_rate_input)
        self.sofm_classification_param_widget_left_layout.addWidget(self.classification_muap_averaging_amount_label)
        self.sofm_classification_param_widget_left_layout.addWidget(self.classification_muap_averaging_amount_input)
        self.sofm_classification_param_widget_right_layout.addWidget(self.perform_sofm_learning_checkbox)
        self.sofm_classification_param_widget_right_layout.addWidget(self.perform_lvq_learning_checkbox)
        self.sofm_classification_param_widget_right_layout.addWidget(self.perform_muap_class_averaging_checkbox)
        self.sofm_classification_param_widget_right_layout.addWidget(self.perform_muap_decomposition_checkbox)
        self.save_param_button = QPushButton('Save Parameters', self)
        self.save_param_button.clicked.connect(self.save_param)

        self.layout.addWidget(self.ann_param_widget)
        self.layout.addWidget(self.lvq_param_widget)
        self.layout.addWidget(self.sofm_classification_param_widget)
        self.layout.addWidget(self.save_param_button)
        self.setLayout(self.layout)
        self.display_param()

    def reset_default(self):
        self.ann_init_weight = 0.0001
        self.ann_init_midweight_times = 0.01
        self.learning_rate = 1
        self.lvq_gaussian_hk = 0.2
        self.lvq_gaussina_thresh = 0.005
        self.lvq_second_winner_adapt_thresh = 0.1
        self.lvq_gaussian_hk_adjust_thresh = 0.01
        self.classification_superimposition_thresh = 0.2
        self.classification_muap_averaging_amount = 3

    def display_param(self):
        self.ann_init_weight_input.setText(str(self.ann_init_weight))
        self.ann_init_midweight_input.setText(str(self.ann_init_midweight_times))
        self.classification_learning_rate_input.setText(str(self.learning_rate))
        self.lvq_hk_input.setText(str(self.lvq_gaussian_hk))
        self.lvq_hk_thresh_input.setText(str(self.lvq_gaussina_thresh))
        self.lvq_second_winner_adapt_thresh_input.setText(str(self.lvq_second_winner_adapt_thresh))
        self.lvq_hk_adjust_thresh_input.setText(str(self.lvq_gaussian_hk_adjust_thresh))
        self.classification_superimposition_thresh_input.setText(str(self.classification_superimposition_thresh))
        self.classification_muap_averaging_amount_input.setText(str(self.classification_muap_averaging_amount))

    def save_param(self):
        init_w = float(self.ann_init_weight_input.text())
        init_mw = float(self.ann_init_midweight_input.text())
        lr = float(self.classification_learning_rate_input.text())
        lghk = float(self.lvq_hk_input.text())
        lghkt = float(self.lvq_hk_thresh_input.text())
        lst = float(self.lvq_second_winner_adapt_thresh_input.text())
        lghka = float(self.lvq_hk_adjust_thresh_input.text())
        cs = float(self.classification_superimposition_thresh_input.text())
        aa = int(self.classification_muap_averaging_amount)

        if init_w <= 0 or init_w > 1:
            init_w = 0.0001
        if init_mw <= 0 or init_mw > 1:
            init_mw = 0.01
        if lr <= 0 or lr > 1:
            lr = 1
        if lghk <= 0 or lghk >1:
            lghk = 0.2
        if lghkt <= 0 or lghkt > 1:
            lghkt = 0.005
        if lst <= 0 or lst > 1:
            lst = 0.1
        if lghka <= 0 or lghka > 1:
            lghka = 0.01
        if cs <= 0 or cs > 1:
            cs = 0.2
        if aa < 0:
            aa = 3
        self.ann_init_weight = init_w
        self.ann_init_midweight_times = init_mw
        self.learning_rate = lr
        self.lvq_gaussian_hk = lghk
        self.lvq_gaussina_thresh = lghkt
        self.lvq_second_winner_adapt_thresh = lst
        self.lvq_gaussian_hk_adjust_thresh = lghka
        self.classification_superimposition_thresh = cs
        self.classification_muap_averaging_amount = aa
        self.display_param()

    def get_ann_weights(self, input_nodes, output_nodes, muaps):
        muap_size = [output_nodes, input_nodes]
        weights = np.zeros((muap_size[0], muap_size[1]), dtype=np.float64)
        weights += self.ann_init_weight
        weights[int(muap_size[0] / 2) + 1, :] = self.ann_init_midweight_times * muaps[0]
        return weights
    def sofm_learning_phase(self, muaps, input_nodes, output_nodes, weights, epochs=1):
        muap_size = [output_nodes, input_nodes]
        node_winning_amount = [0 for _ in range(muap_size[0])]
        for _ in range(epochs):
            for i in range(len(muaps)):
                print('....Current MUAP shape: ' + str(len(muaps[i])))
                print('....Current Weight Shape: ' + str(len(weights[0])))
                distance_k = [np.sum(np.square(np.asarray(muaps[i]) - weights[j])) for j in range(muap_size[0])]
                winner_node = np.argmin(distance_k)
                node_winning_amount[winner_node] += 1
                print('....Min Distance for MUAP No. ' + str(i) + ': ' + str(distance_k[winner_node]))

                # Weight adjustment for each output node - LEARNING PHASE 1
                g = self.learning_rate  # 0 < g < 1
                t = 0
                for k in range(len(weights)):
                    t = t + 1  # No.of iteration: starts from 1
                    print('........Adjusting weights for outputput node ' + str(k))

                    if (k == winner_node and node_winning_amount[k] == 1):
                        gaussian_hk = 1
                    elif node_winning_amount[k] == 0:
                        gaussian_hk = 1
                    else:
                        gaussian_hk = g * math.exp(-(k - winner_node) ** 2 * t / 2) / math.sqrt(node_winning_amount[k])
                    print('........Gaussian Value: ' + str(gaussian_hk))
                    if gaussian_hk >= 0.005:
                        print('........Adapting Weights')
                        for x in range(1, weights.shape[1]):
                            weights[k][x] = weights[k][x - 1] + gaussian_hk * (muaps[i][x] - weights[k][x - 1])
                print('-----------------------------------------------------------------\n')
        return weights

    def lvq_learning_phase(self, muaps, input_nodes, output_nodes, weights, epochs=1):
        muap_size = [output_nodes, input_nodes]
        node_winning_amount = [0 for _ in range(muap_size[0])]
        lvq_gaussian_hk = self.lvq_gaussian_hk
        for _ in range(epochs):
            for i in range(len(muaps)):
                print('....Current MUAP shape: ' + str(len(muaps[i])))
                print('....Current Weight Shape: ' + str(len(weights[0])))
                distance_k = [np.sum(np.square(np.asarray(muaps[i]) - weights[j])) for j in range(muap_size[0])]
                winner_node = np.argmin(distance_k)
                second_winner_node = np.argpartition(distance_k, 1)[1]
                node_winning_amount[winner_node] += 1
                node_winning_amount[second_winner_node] += 1
                print('....Min Distance for MUAP No. ' + str(i) + ': ' + str(distance_k[winner_node]))

                # Weight adjustment for each output node - LEARNING PHASE 1
                g = 1  # 0 < g < 1
                t = i + 1  # No.of iteration: starts from 1
                #rint('........Adjusting weights for outputput node ' + str(k))

                print('........Gaussian Value: ' + str(lvq_gaussian_hk))
                for x in range(1, weights.shape[1]):
                    weights[winner_node][x] = weights[winner_node][x - 1] + lvq_gaussian_hk * (
                            muaps[i][x] - weights[winner_node][x - 1])
                    weights[second_winner_node][x] = weights[second_winner_node][x - 1] \
                                                     - self.lvq_second_winner_adapt_thresh * (distance_k[winner_node] / distance_k[
                        second_winner_node]) \
                                                     * lvq_gaussian_hk * (
                                                             muaps[i][x] - weights[second_winner_node][x - 1])
                lvq_gaussian_hk = self.lvq_gaussian_hk - self.lvq_gaussian_hk_adjust_thresh * node_winning_amount[winner_node]
                if lvq_gaussian_hk < 0:
                    lvq_gaussian_hk = 0
        return weights

    def sofm_classification(self, muaps, input_nodes, output_nodes, weights):
        muap_size = [output_nodes, input_nodes]
        muap_classification_output = [-1 for _ in range(len(muaps))]  # 0 for actual muap and 1 for superimposed muap
        muap_classification_class = [-1 for _ in range(len(muaps))]  # Classification class for MUAP

        for i in range(len(muaps)):
            distance_k = [np.sum(np.square(np.asarray(muaps[i]) - weights[j])) for j in range(muap_size[0])]
            winner_node = np.argmin(distance_k)
            muap_classification_class[i] = winner_node

            length_kw = np.sum(weights[winner_node] ** 2)
            print(
                'Classification Threshold for MUAP No. ' + str(i + 1) + ': ' + str(distance_k[winner_node] / length_kw))
            if distance_k[winner_node] / length_kw < self.classification_superimposition_thresh:
                muap_classification_output[i] = 0
            else:
                muap_classification_output[i] = 1
        return muap_classification_class, muap_classification_output

    def average_muap_templates(self, muaps, muap_class, muap_superimposition, weights, input_nodes, output_nodes, firing_time=[]):
        muaps_detected = [[] for _ in range(output_nodes)]
        for i in range(len(muaps)):
            if muap_superimposition[i] == 0:
                muaps_detected[muap_class[i]].append(i)
        new_muaps = []
        new_muap_class = []
        new_muap_superimposition = []
        if len(firing_time) > 0:
            new_firing_time = []
        for i in range(len(muaps_detected)):
            if len(muaps_detected[i]) > self.classification_muap_averaging_amount:
                average_muap = np.zeros((input_nodes), dtype=np.float64)
                ft = []
                for j in range(len(muaps_detected[i])):
                    average_muap = average_muap + np.asarray(muaps[muaps_detected[i][j]])
                    ft = ft + firing_time[muaps_detected[i][j]]
                average_muap = average_muap/len(muaps_detected[i])
                avg_muap_class, avg_muap_super = self.sofm_classification([average_muap.tolist()], input_nodes, output_nodes, weights)
                if avg_muap_super[0] == 0:
                    new_muaps.append(average_muap.tolist())
                    new_muap_class.append(avg_muap_class[0])
                    new_muap_superimposition.append(avg_muap_super[0])
                    new_firing_time.append(ft)
                else:
                    for j in range(len(muaps_detected[i])):
                        new_muaps.append(muaps[muaps_detected[i][j]])
                        new_muap_class.append(muap_class[muaps_detected[i][j]])
                        new_muap_superimposition.append(1)
                        new_firing_time.append(firing_time[muaps_detected[i][j]])
            else:
                for j in range(len(muaps_detected[i])):
                    new_muaps.append(muaps[muaps_detected[i][j]])
                    new_muap_class.append(muap_class[muaps_detected[i][j]])
                    new_muap_superimposition.append(muap_superimposition[muaps_detected[i][j]])
                    new_firing_time.append(firing_time[muaps_detected[i][j]])
        for i in range(len(muaps)):
            if muap_superimposition[i] == 1:
                new_muaps.append(muaps[i])
                new_muap_class.append(muap_class[i])
                new_muap_superimposition.append(muap_superimposition[i])
                new_firing_time.append(firing_time[i])

        return new_muaps, new_muap_class, new_muap_superimposition, new_firing_time

def perform_emg_decomposition(waveforms, waveform_classes, waveform_superimposition, firing_time):

    waveforms = np.random.randint(0, 10, 16).reshape((4, 4))
    waveform_classes = [0, 1, 2, 1]
    waveform_superimposition = [0, 1, 1, 0]
    firing_time = [[1], [3, 7, 4], [2, 5], [8]]
    print('Waveform: ' + str(waveforms))
    actual_muaps = []
    superimposed_muaps = []
    for i in range(len(waveforms)):
        if waveform_superimposition[i] == 0:
            actual_muaps.append([waveforms[i], waveform_classes[i], firing_time[i]])
        else:
            superimposed_muaps.append([waveforms[i], waveform_classes[i], firing_time[i]])
    print('Actual MUAPS: ' + str(actual_muaps))
    print('Superimposed MUAPS: ' + str(superimposed_muaps))

    reduce_amplitude = 1/15
    for i in range(len(actual_muaps)):
        muap_amplitude = np.amax(actual_muaps[i][0]) - np.amin(actual_muaps[i][0])
        print('MUAP amplitude: ' + str(muap_amplitude))
        trim_length = int(math.ceil(reduce_amplitude*muap_amplitude))
        print('Trim length: ' + str(trim_length))
        actual_muap = actual_muaps[i][0]
        trimmed_muap = []
        for j in range(len(actual_muap)/2, len(actual_muap)):
            if actual_muap[j] > trim_length:
                trimmed_muap.append(actual_muap[j])
            else:
                break


        print('Trimmed MUAP: ' + str(trimmed_muap))
perform_emg_decomposition(None, None, None, None)