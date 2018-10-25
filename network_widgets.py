import sys, math, pdb, collections, _collections
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

import random
from win32api import GetSystemMetrics

from scipy.signal import butter, lfilter, spectrogram, find_peaks, iirfilter
import  scipy.fftpack as fftpack
from skimage.feature import peak_local_max
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class ClassifierWidget:
    def __init__(self, network_name, network_number, network_widgets, parent):
        super().__init__()
        self.network_name = network_name
        self.network_number = network_number
        self.network_widgets = network_widgets
        self.parent = parent

        self.main_scroller = QScrollArea(self.parent)
        self.main_widget = QGroupBox(str(network_name), self.main_scroller)
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.main_scroller.setWidget(self.main_widget)
        self.main_scroller.setWidgetResizable(True)

        self.graph_widget = QGroupBox('Graphs', self.parent)
        self.graph_widget_layout = QVBoxLayout()
        self.graph_widget.setLayout(self.graph_widget_layout)
        self.info_widget = QGroupBox('Information', self.parent)
        self.info_widget_layout = QVBoxLayout()
        self.info_widget.setLayout(self.info_widget_layout)
        self.main_layout.addWidget(self.graph_widget)
        self.main_layout.addWidget(self.info_widget)

        self.graphs = collections.OrderedDict()
        self.graphs['accuracy'] = None
        self.graphs['average accuracy'] = None
        self.graphs['data'] = None

        graph_code = 100
        for k, v in self.graphs.items():
            figure = plt.figure(int(network_number) + graph_code)
            figure.suptitle(k.upper())
            graph_axis = figure.add_subplot(1, 1, 1)
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, self.parent)
            self.graph_widget_layout.addWidget(canvas)
            self.graph_widget_layout.addWidget(toolbar)

            self.graphs[k] = {
                "fig": figure,
                "axis": graph_axis,
                "canvas": canvas,
                "toolbar": toolbar
            }
            graph_code += 100
        children_widget_layout = QHBoxLayout
        status_widget = QComboBox(self.parent)
        status_widget.addItem('Deactive')
        status_widget.addItem('Active')
        self.info_type = {
            'classifier': {
                'children': {
                    'status': {
                        'value': status_widget,
                        'label': QLabel('Status: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'name': {
                        'value': QLabel(str(network_name).upper(), self.parent),
                        'label': QLabel('Name: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'x_feature': {
                        'value': QLineEdit(self.parent),
                        'label': QLabel('X-axis Feature: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'y_feature': {
                        'value': QLineEdit(self.parent),
                        'label': QLabel('Y-axis Feature: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'outputs': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Total Output Classes: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'output_labels': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Output Labels: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    }
                },
                'widget': QGroupBox('Network Info', self.parent),
                'layout': QVBoxLayout()
            },
            'training': {
                'children': {
                    'train_data_size': {
                        'value': QLineEdit(self.parent),
                        'label': QLabel('Train Data Size: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'test_data_size': {
                        'value': QLineEdit(self.parent),
                        'label': QLabel('Name: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'accuracy': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Accuracy: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'loss': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Loss: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'total_data': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Total data loaded: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    }
                },
                'widget': QGroupBox('Training Info', self.parent),
                'layout': QVBoxLayout()
            },
            'prediction': {
                'children': {
                    'input_path': {
                        'value': QLineEdit(self.parent),
                        'label': QLabel('Input Path: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'output_class': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Prediction Class: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'output_confidence': {
                        'value': QLabel("N/A", self.parent),
                        'label': QLabel('Prediction Confidence: ', self.parent),
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    },
                    'button': {
                        'value': QPushButton('Predict', self.parent),
                        'label': None,
                        'widget': QGroupBox('', self.parent),
                        'layout': children_widget_layout()
                    }
                },
                'widget': QGroupBox('Prediction Info', self.parent),
                'layout': QVBoxLayout()
            }
        }
        self.info_type['classifier']['children'].update(self.network_widgets)
        for info in sorted(self.info_type):
            self.info_type[info]['widget'].setLayout(self.info_type[info]['layout'])
            for val in sorted(self.info_type[info]['children']):
                self.info_type[info]['children'][val]['widget'].setLayout(self.info_type[info]['children'][val]['layout'])
                if not (self.info_type[info]['children'][val]['label'] is None):
                    self.info_type[info]['children'][val]['layout'].addWidget(self.info_type[info]['children'][val]['label'])
                if not (self.info_type[info]['children'][val]['value'] is None):
                    self.info_type[info]['children'][val]['layout'].addWidget(self.info_type[info]['children'][val]['value'])
                self.info_type[info]['layout'].addWidget(self.info_type[info]['children'][val]['widget'])
            self.info_widget_layout.addWidget(self.info_type[info]['widget'])

class Classifier(ABC):
    def __init__(self, classifier_name, classifier_number, view_parent):
        super.__init__()

    @abstractmethod
    def create_classifier(self):
        pass
    @abstractmethod
    def train_classifier(self, X, y, shuffle=True):
        pass
    @abstractmethod
    def predict_classifier(self):
        pass

class KNearestClassifier(Classifier):
    def __init__(self, classifier_name, classifier_number, view_parent):
        self.classifier_object = KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=30, p=2)
        self.neighbors_inp = QLineEdit(view_parent)
        self.neighbors_inp.setText('3')
        self.algorithms = {
            'Most Appropriate': 'auto',
            'Ball Tree': 'ball_tree',
            'KD Tree': 'kd_tree',
            'Brute Force Search': 'brute'
        }
        self.algorithm_input = QComboBox(view_parent)
        for k in self.algorithms:
            self.algorithm_input.addItem(k)

        self.leaf_size_inp = QLineEdit(view_parent)
        self.leaf_size_inp.setText(str(30))

        self.power_param_inp = QLineEdit(view_parent)
        self.power_param_inp.setText(str(2))

        self.input_dataset = []
        self.network_widgets = {
            'neighbors': {
                'value': self.neighbors_inp,
                'label': QLabel('Neighbors: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'algorithm': {
                'value': self.algorithm_input,
                'label': QLabel('Algorithm: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'leaf_size': {
                'value': self.leaf_size_inp,
                'label': QLabel('Leaf Size: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'power_param': {
                'value': self.power_param_inp,
                'label': QLabel('Power Parameter: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            }
        }
        self.view = ClassifierWidget(classifier_name, classifier_number, self.network_widgets, view_parent)


    def create_classifier(self):
        neighb = int(self.neighbors_inp.text())
        alg = str(self.algorithm_input.currentText())
        lsize = int(self.leaf_size_inp.text())
        pparam = int(self.power_param_inp.text())
        self.classifier_object = KNeighborsClassifier(n_neighbors=neighb, algorithm=self.algorithms[alg], leaf_size=lsize, p=pparam)

    def train_classifier(self, X, y, shuffle=True):

        total_data_len = len(self.input_dataset)
        if len(str(self.view.info_type['training']['children']['train_data_size']['value'].text())) > 0:
            train_size = int(self.view.info_type['training']['children']['train_data_size']['value'].text())
        else:
            train_size = 5
        if len(str(self.view.info_type['training']['children']['test_data_size']['value'].text())) > 0:
            test_size = int(self.view.info_type['training']['children']['test_data_size']['value'].text())
        else:
            test_size = 1
        test_perc = test_size/(test_size+train_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_perc, random_state = 42, shuffle=shuffle)
        self.classifier_object.fit(X_train, y_train)
        accuracy = self.classifier_object.score(X_test, y_test)
        loss = 1 - accuracy
        return accuracy, loss

    def predict_classifier(self):
        X = np.load(str(self.view.info_type['prediction']['children']['input_path']['value'].text()))
        prediction = self.classifier_object.predict(X)
        output_class = []
        output_confidence = []
        for i in range(len(prediction)):
            index = np.argmax(prediction[i])
            output_class.append(index)
            output_confidence.append(prediction[i][index])
        return output_class, output_confidence

class SVMCLassifier(Classifier):
    def __init__(self, classifier_name, classifier_number, view_parent):
        self.classifier_object = SVC(kernel='rbf', degree=3, gamma='auto', probability=False, decision_function_shape='ovr')

        self.kernels = {
            'Linear': 'linear', 'Polynomial': 'poly', 'RBF': 'rbf', 'Sigmoid': 'sigmoid'
        }
        self.kernel_input = QComboBox(view_parent)
        for k in self.kernels:
            self.kernel_input.addItem(self.kernels[k])
        self.degree_inp = QLineEdit(view_parent)
        self.degree_inp.setText('3')
        self.gammas = {'1/n Features': 'auto'}
        self.gamma_input = QComboBox(view_parent)
        for k in self.gammas:
            self.gamma_input.addItem(self.gammas[k])
        self.probability_inp = QComboBox(view_parent)
        self.probability_inp.addItem('True')
        self.probability_inp.addItem('False')
        self.tolerance_input = QLineEdit(view_parent)
        self.tolerance_input.setText('1e-3')
        self.decision_functions = {
            'One Versus One': 'ovo',
            'One Versus Rest': 'ovr'
        }
        self.decision_func_inp = QComboBox(view_parent)
        for k in self.decision_functions:
            self.decision_func_inp.addItem(self.decision_functions[k])



        self.input_dataset = []
        self.network_widgets = {
            'kernel': {
                'value': self.kernel_input,
                'label': QLabel('Kernel: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'degree': {
                'value': self.degree_inp,
                'label': QLabel('Degree of Polynomial Kernel Function: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'gamma': {
                'value': self.gamma_input,
                'label': QLabel('Kernel Coefficient: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'probability': {
                'value': self.probability_inp,
                'label': QLabel('Probability Estitmate: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'decision_function': {
                'value': self.decision_func_inp,
                'label': QLabel('Decision Function: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            }
        }
        self.view = ClassifierWidget(classifier_name, classifier_number, self.network_widgets, view_parent)
    def create_classifier(self):
        kernel = str(self.kernel_input.currentText())
        degree = int(self.degree_inp.text())
        gamma = str(self.gamma_input.currentText())
        prob = str(self.probability_inp.currentText())
        dec_fn = str(self.decision_func_inp.currentText())
        self.classifier_object = SVC(kernel=kernel, degree=degree, gamma=gamma, probability=prob=='True',
                                     decision_function_shape=dec_fn)

    def train_classifier(self, X, y, shuffle=True):

        total_data_len = len(self.input_dataset)
        if len(str(self.view.info_type['training']['children']['train_data_size']['value'].text())) > 0:
            train_size = int(self.view.info_type['training']['children']['train_data_size']['value'].text())
        else:
            train_size = 5
        if len(str(self.view.info_type['training']['children']['test_data_size']['value'].text())) > 0:
            test_size = int(self.view.info_type['training']['children']['test_data_size']['value'].text())
        else:
            test_size = 1
        test_perc = test_size / (test_size + train_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=42, shuffle=shuffle)
        self.classifier_object.fit(X_train, y_train)
        accuracy = self.classifier_object.score(X_test, y_test)
        loss = 1 - accuracy
        return accuracy, loss
    def predict_classifier(self):
        X = np.load(str(self.view.info_type['prediction']['children']['input_path']['value'].text()))
        prediction = self.classifier_object.predict(X)
        output_class = []
        output_confidence = []
        for i in range(len(prediction)):
            index = np.argmax(prediction[i])
            output_class.append(index)
            output_confidence.append(prediction[i][index])
        return output_class, output_confidence

class RForestCLassifier(Classifier):
    def __init__(self, classifier_name, classifier_number, view_parent):
        self.classifier_object = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2,
                                                        max_features='auto', warm_start=False)
        self.trees_input = QLineEdit(view_parent)
        self.trees_input.setText('10')
        self.max_depth_input = QLineEdit(view_parent)
        self.max_depth_input.setText('None')
        self.sample_split_input = QLineEdit(view_parent)
        self.sample_split_input.setText('2')
        self.max_features_input = QLineEdit(view_parent)
        self.max_features_input.setText('auto')
        self.warm_start_input = QComboBox(view_parent)
        self.warm_start_input.addItem('True')
        self.warm_start_input.addItem('False')

        self.input_dataset = []
        self.network_widgets = {
            'trees': {
                'value': self.trees_input,
                'label': QLabel('No. of Trees: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'max_depth': {
                'value': self.max_depth_input,
                'label': QLabel('Maximum depth: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'sample_split': {
                'value': self.sample_split_input,
                'label': QLabel('Minimum samples: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'max_features': {
                'value': self.max_features_input,
                'label': QLabel('Split Features: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            },
            'warm_start': {
                'value': self.warm_start_input,
                'label': QLabel('Warm Start: ', view_parent),
                'widget': QGroupBox('', view_parent),
                'layout': QHBoxLayout()
            }
        }
        self.view = ClassifierWidget(classifier_name, classifier_number, self.network_widgets, view_parent)


    def create_classifier(self):
        trees = int(self.trees_input.text())
        max_depth = str(self.max_depth_input.text())
        if max_depth.lower() != 'none':
            max_depth = int(max_depth)
        else:
            max_depth = None
        sample_split = int(self.sample_split_input.text())
        max_features = str(self.max_features_input.text())
        if max_features.lower() != 'auto':
            max_features = int(max_features)
        warm_start = str(self.warm_start_input.currentText())
        self.classifier_object = RandomForestClassifier(n_estimators=trees, max_depth=max_depth, min_samples_split=sample_split,
                                                        max_features=max_features, warm_start=warm_start)

    def train_classifier(self, X, y, shuffle=True):

        total_data_len = len(self.input_dataset)
        if len(str(self.view.info_type['training']['children']['train_data_size']['value'].text())) > 0:
            train_size = int(self.view.info_type['training']['children']['train_data_size']['value'].text())
        else:
            train_size = 5
        if len(str(self.view.info_type['training']['children']['test_data_size']['value'].text())) > 0:
            test_size = int(self.view.info_type['training']['children']['test_data_size']['value'].text())
        else:
            test_size = 1
        test_perc = test_size / (test_size + train_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=42, shuffle=shuffle)
        self.classifier_object.fit(X_train, y_train)
        accuracy = self.classifier_object.score(X_test, y_test)
        loss = 1 - accuracy
        return accuracy, loss
    def predict_classifier(self):
        X = np.load(str(self.view.info_type['prediction']['children']['input_path']['value'].text()))
        prediction = self.classifier_object.predict(X)
        output_class = []
        output_confidence = []
        for i in range(len(prediction)):
            index = np.argmax(prediction[i])
            output_class.append(index)
            output_confidence.append(prediction[i][index])
        return output_class, output_confidence