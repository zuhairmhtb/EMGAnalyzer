# EMGAnalyzer
This is a Python based Software developed for the automation of EMG Signal analysis. The specifications for the  software are as follows:
1. Dataset Structure:
    a. The main dataset directory consists of two folders (train and test) where the 'test' folder contains patient records for testing  the classifier and 'train' folder contains patient record for training the classifier.
    b. Each of the train/test directory consists of three folders(myopathy, ALS and normal) where each folder contains folders for different subjects who falls under the specified group. The record folders of each subject is stored in a folder(patient folder) bearing a unique ID number(e.g. a01_patient, c01_patient, etc.) for each individual patient. Each patient folder can have multiple EMG record folders obtained from the brachial biceps of the subject. Each record folder contains information related to each signal recorded from the specific patient.
    c. Each record folder of the subjects also bear a unique ID number(e.g. N2001A01BB05, N2001A01BB06, etc.) and each folder contains three files. They are 'data.npy' and 'data.hea'.
    d. data.npy: This file contains the EMG signal recorded from an electromyograph. The data is stored as a 'Numpy' one dimensional array where the length of array indicates the number of samples obtained at a specific sampling frequency.
    e. data.hea: This is a WFDB header file that contains all the information regarding subject under investigation and recorded EMG signal from the subject. As for example it contains the sampling frequency and total number of samples obtained from the signal, gender of the subject, period of diagnosis, duration of disease, location of placement of electrode, filters used, level of insertion of needle, etc. The data is stored as a text file (More documentation on WFDB header files: http://www.emglab.net/emglab/Tutorials/WFDB.html).
2. Required Software Packages:
   PyQt5, pywt, Matplotlib, Scipy
3. Software Description:
