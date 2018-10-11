import sys, math, pdb, collections, queue
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

from scipy.signal import butter, lfilter, spectrogram, find_peaks, iirfilter, correlate
import  scipy.fftpack as fftpack
from skimage.feature import peak_local_max


class ClassificationControlWindow(QWidget):
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
        self.save_muap_waveform_file = "muap_waveforms"
        self.save_muap_classes_file = "muap_output_classes"
        self.save_muap_superimposition_file = "muap_superimposition_classes"
        self.save_muap_firing_time_file = "muap_firing_time"
        self.save_muap_firing_table_file = "muap_firing_table"



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

        self.save_muap_waveform_checkbox = QCheckBox('Save MUAP Waveforms', self)
        self.save_muap_classes_checkbox = QCheckBox('Save MUAP Classes', self)
        self.save_muap_superimposition_checkbox = QCheckBox('Save Superimposition/Actual Classes', self)
        self.save_muap_firing_time_checkbox = QCheckBox('Save Firing Time', self)
        self.save_muap_firing_table_checkbox = QCheckBox('Save Firing Table', self)
        self.save_param_checkbox_widget = QGroupBox('Save Parameters', self)
        self.save_param_checkbox_widget_layout = QHBoxLayout()
        self.save_param_checkbox_widget.setLayout(self.save_param_checkbox_widget_layout)
        self.save_param_checkbox_widget_layout.addWidget(self.save_muap_waveform_checkbox)
        self.save_param_checkbox_widget_layout.addWidget(self.save_muap_classes_checkbox)
        self.save_param_checkbox_widget_layout.addWidget(self.save_muap_superimposition_checkbox)
        self.save_param_checkbox_widget_layout.addWidget(self.save_muap_firing_table_checkbox)
        self.save_param_checkbox_widget_layout.addWidget(self.save_muap_firing_time_checkbox)

        self.save_param_button = QPushButton('Save Parameters', self)
        self.save_param_button.clicked.connect(self.save_param)


        self.layout.addWidget(self.ann_param_widget)
        self.layout.addWidget(self.lvq_param_widget)
        self.layout.addWidget(self.sofm_classification_param_widget)
        self.layout.addWidget(self.save_param_checkbox_widget)
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


    def perform_emg_decomposition(self, waveforms, waveform_classes, waveform_superimposition, firing_time,
                                      calculate_endpoints=False, pearson_correlate=True, plot=False):
        print('Waveform: ' + str(len(waveforms)))
        max_residue_amp = 30
        # Separate Actual and Superimposed Waveform
        actual_muaps = []
        superimposed_muaps = []
        residue_superimposed_muaps = []
        for i in range(len(waveforms)):

            if waveform_superimposition[i] == 0:
                actual_muaps.append([waveforms[i], waveform_classes[i], firing_time[i]])
            else:
                superimposed_muaps.append([waveforms[i], waveform_classes[i], firing_time[i]])
        print('Actual MUAPS: ' + str(len(actual_muaps)))
        print('Superimposed MUAPS: ' + str(len(superimposed_muaps)))

        # Create a queue that will hold the superimposed waveforms that needs to be decomposed
        superimposed_queue = queue.Queue()
        for i in range(len(superimposed_muaps)):
            superimposed_queue.put(list(superimposed_muaps[i]).copy())
        # For each superimposed waveform perform the following tasks
        cur = 0
        while not superimposed_queue.empty():
            try:
                print('Superimposed waveform left: ' + str(superimposed_queue.qsize()))
                cur += 1
                smuap = superimposed_queue.get()
                if len(smuap[0] < len(actual_muaps[0][0])):
                    smuap[0] = list(smuap[0]) + [smuap[0][-1]] * (len(actual_muaps[0][0]) - len(smuap[0]))
                # Cross correlate each reduced MUAP with the superimposed waveform x and find the best matching point
                # i.e. the points where crosscorrelation takes the maximum value

                print('Crosscorrelating superimposed waveform of length: ' + str(len(smuap[0])))

                best_matching_points = []  # Best matching point and maximum correlation coefficient for each MUAP with the superimposed waveform
                nds = []  # Normalized Euclidean distance for each matching pair
                ads = []  # Average Area Difference for each matching pair
                ths = []  # Varying Threshold for each matching pair
                adjusted_waveforms = []
                for j in range(len(actual_muaps)):
                    print("Correlating with reduced MUAP of length: " + str(len(actual_muaps[j][0])))

                    if pearson_correlate:
                        x = np.asarray(smuap[0]).astype(np.float64) / np.std(smuap[0])
                        y = np.asarray(actual_muaps[j][0]).astype(np.float64) / np.std(actual_muaps[j][0])
                    else:
                        x = np.asarray(smuap[0]).astype(np.float64)
                        y = np.asarray(actual_muaps[j][0]).astype(np.float64)
                    correlation = correlate(x, y)
                    print("Cross correlation shape: " + str(len(correlation)))
                    print('Maximum Coefficient: ' + str(np.amax(correlation)) + " at index " + str(
                        np.argmax(correlation)))
                    highest_cor_ind = np.argmax(correlation)
                    best_matching_points.append([correlation[highest_cor_ind], highest_cor_ind])
                    # Calculate normalized Euclidean Distance, Average Area Difference and Varying Threshold for the
                    # matching pair i.e the actual muap and portion of the superimposed waveform that has the highest
                    # similarity with the muap

                    if highest_cor_ind < len(smuap[0]) - 1:
                        print('Less')
                        smuap_start = 0
                        smuap_end = highest_cor_ind + 1
                        muap_start = len(actual_muaps[j][0]) - highest_cor_ind - 1
                        muap_end = len(actual_muaps[j][0])

                    elif highest_cor_ind == len(smuap[0]) - 1:
                        smuap_start = 0
                        smuap_end = len(smuap[0])
                        muap_start = 0
                        muap_end = len(actual_muaps[j][0])
                    else:
                        print('More')
                        smuap_start = highest_cor_ind - (len(smuap[0]) - 1)
                        smuap_end = len(smuap[0])
                        muap_start = 0
                        muap_end = len(actual_muaps[j][0]) - (highest_cor_ind - len(smuap[0]) + 1)
                    print(str(cur) + ', ' + str(j))

                    adjusted_superimposed = np.asarray(smuap[0])[smuap_start:smuap_end]
                    adjusted_muap = np.asarray(actual_muaps[j][0])[muap_start:muap_end]
                    adjusted_waveforms.append(
                        [adjusted_superimposed, adjusted_muap, [smuap_start, smuap_end], [muap_start, muap_end]])

                    nd = np.sum(np.subtract(adjusted_superimposed, adjusted_muap) ** 2) / np.sum(
                        np.multiply(adjusted_muap, adjusted_muap))
                    nds.append(nd)
                    # Average Area Difference
                    ad = np.sum(np.abs(np.subtract(adjusted_superimposed, adjusted_muap))) / len(adjusted_muap)
                    ads.append(ad)
                    # Varying Threshold
                    threshold_const_a = 0.5
                    threshold_const_b = 4
                    th = threshold_const_b + threshold_const_a * np.sum(np.abs(adjusted_muap)) / len(adjusted_muap)
                    ths.append(th)

                # Matching pair with minimum classification coefficient

                nd_thresh1 = 0.2
                nd_thresh2 = 0.5

                best_matching_muap = -1
                min_coeff_thresh = sys.maxsize
                for j in range(len(best_matching_points)):
                    if (nds[j] < nd_thresh1 or (ads[j] < ths[j] and nds[j] < nd_thresh2)):
                        class_coeff = (nds[j] * ads[j]) / (ths[j] * len(adjusted_muap))
                        if class_coeff < min_coeff_thresh:
                            best_matching_muap = j
                            min_coeff_thresh = class_coeff

                if best_matching_muap >= 0:
                    # A MUAP Class is identified for the superimposed waveform.
                    # Decompose the superimposed waveform from the MUAP class.
                    class_smuap = list(smuap[0])
                    class_muap = list(actual_muaps[best_matching_muap][0])
                    residue_signal = []
                    highest_cor_ind = best_matching_points[best_matching_muap][1]
                    # Pad MUAP and SMUAP arrays with zero in order to make them equal length and subtract
                    if highest_cor_ind < len(smuap[0]):
                        class_smuap = [0] * adjusted_waveforms[best_matching_muap][3][0] + class_smuap
                        class_muap = class_muap + [0] * adjusted_waveforms[best_matching_muap][3][0]
                        class_smuap_start = adjusted_waveforms[best_matching_muap][3][0]
                        class_smuap_end = adjusted_waveforms[best_matching_muap][3][0] + len(smuap[0])
                    elif highest_cor_ind > len(smuap[0]):
                        class_muap = [0] * adjusted_waveforms[best_matching_muap][2][0] + class_muap
                        class_smuap = class_smuap + [0] * adjusted_waveforms[best_matching_muap][2][0]
                        class_smuap_start = 0
                        class_smuap_end = len(smuap[0])

                    # Calculate residue signal
                    residue_signal = np.subtract(class_smuap, class_muap)
                    max_residue_amp = 30  # uV
                    if np.amax(residue_signal[class_smuap_start:class_smuap_end]) < max_residue_amp:
                        # If the max amplitude of residue signal is greater than threshold, then feed it back to the
                        # queue for further decomposition
                        smuap[0] = residue_signal[class_smuap_start:class_smuap_end]
                        superimposed_queue.put(smuap)
                    else:
                        # Else add it to the list of decomposed residue signal
                        residue_superimposed_muaps.append(smuap)
                    # Update Firing time of the Best Matching MUAP with the firing time of the superimposed signal
                    actual_muaps[best_matching_muap][2] += smuap[2]
                    if plot:
                        plt.subplot(3, 1, 1)
                        plt.title('Best Matching MUAP: Cross Correlation Index: ' + str(highest_cor_ind))
                        plt.plot(smuap[0], label='SMUAP')
                        plt.plot(np.arange(highest_cor_ind - len(actual_muaps[best_matching_muap][0]), highest_cor_ind),
                                 actual_muaps[best_matching_muap][0],
                                 label='MUAP')
                        plt.plot(residue_signal[class_smuap_start:class_smuap_end], label='Residue')
                        plt.grid()
                        plt.legend()
                        plt.subplot(3, 1, 2)
                        plt.title(
                            'Nd: ' + str(nds[best_matching_muap]) + ', Ad: ' + str(ads[best_matching_muap]) +
                            ', Th: ' + str(ths[best_matching_muap]) + ', Coeff: ' + str(min_coeff_thresh))
                        plt.plot(adjusted_waveforms[best_matching_muap][0],
                                 label='SMUAP: ' + str(
                                     adjusted_waveforms[best_matching_muap][2][1] -
                                     adjusted_waveforms[best_matching_muap][2][
                                         0]))
                        plt.plot(adjusted_waveforms[best_matching_muap][1], label='MUAP: ' + str(
                            adjusted_waveforms[best_matching_muap][3][1] - adjusted_waveforms[best_matching_muap][3][
                                0]))
                        plt.grid()
                        plt.legend()

                        plt.subplot(3, 1, 3)
                        plt.title('Decomposition')
                        plt.plot(class_smuap, label='SMUAP')
                        plt.plot(class_muap, label='MUAP')
                        plt.plot(residue_signal, label='Residue')
                        plt.grid()
                        plt.legend()

                        plt.show()


                else:
                    print("No Class identified for the superimposed waveform. Removing it from the list")
                    residue_superimposed_muaps.append(smuap)
            except:
                print("Error occured here")


        return actual_muaps, residue_superimposed_muaps




def perform_cross_correlation(x, y):
    x = [3,2,1]
    y = [2,1]
    if x < y:
        temp = y
        y = x
        x = temp
    result = []
    pad_half = 0
    if len(y) > 1:
        pad_half = len(y)-1
        x = [0]*pad_half + x + [0]*pad_half
    print('Padded X: ' + str(x))
    for i in range(len(x)-len(y)+1):
        result.append(np.sum(np.multiply(x[i:i+len(y)], y)))
    print('Result: ' + str(result))



dir = "D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\train\\als\\a01_patient\\N2001A01BB04\\"
a = np.load(dir+"muap_waveforms.npy")
b = np.load(dir+"muap_output_classes.npy")
c = np.load(dir+"muap_superimposition_classes.npy")
d = np.load(dir+"muap_firing_time.npy")
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
#ClassificationControlWindow.perform_emg_decomposition(None, a, b, c, d, plot=True)
#perform_cross_correlation(1, 2)