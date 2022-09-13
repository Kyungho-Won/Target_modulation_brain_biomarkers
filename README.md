# **Target Modulation & Brain Biomarkers**
### Kyungho Won, Sunghan Lee, Daeun Gwon, Sooyeon Kim
![alt text](https://github.com/KyunghoWon-GIST/Target_modulation_brain_biomarkers/blob/main/image_readme.png)

### ■ Stimulation Presentation Task
![alt text](https://github.com/KyunghoWon-GIST/Target_modulation_brain_biomarkers/blob/main/Stimulation_presentation_Task/images/image_mlapp.png)
#### It includes MATLAB, C scripts, and MATLAB based GUI application for basic stimulation presentation and feature extraction during motor imagery task

#### - External dependancies
For pre-processing and display scalp topography, MATLAB based EEGLAB (https://sccn.ucsd.edu/eeglab/index.php) and FieldTrip (https://www.fieldtriptoolbox.org/) toolbox are required. 

#### - Structure
```
>> Stimuluation_presentation_Task % 
./OpenViBE_scenario % OpenViBE scenario (*.MXS) files synchrnoized to mlapp. 
./functions
./images
./images_MI % Stimulation images
./matlabox_sync_with_openViBE % MATLAB scripts within OpenViBE onlien signal processing pieline
./Brain_Cognition_Modulation_Task.mlapp
```
#### - Mlapp tabs
```
- CONFIG % hyper parameter settings for motor imagery task and play (run) the task
- TRAIN % train EEG feature using EEG data collected during offline motor imagery task and export classifier weights in *mat format
```

### ■ Extract BioMarker VI Task

#### It includes MATLAB scripts and deep learning models to extract features and classify user intention from visual imagery (VI) task

#### - Structure
```
./EEGNet-maser
./MATLAB
./MultiRocket-main
```

### ■ Extraction Biomarker of Motor Imagery Task
>This is the AI-based Brain Signal Processing Algorithm - Motor Function project. They consist of EEG signal classification and dataset loading code. It is the Convolutional Neural Network (CNN)-Long Short-Term Memory (LSTM) models for signal processing and classification on motor imagery, written in PyTorch and Tensorflow.  It also contains code for loading multiple datasets written in MATLAB.

This is a schematic diagram of the proposed CNN-LSTM model. We propose a new model by referring to the following paper: Zhang, Ruilong, et al. "Hybrid deep neural network using transfer learning for EEG motor imagery decoding." Biomedical Signal Processing and Control 63 (2021): 102144.
![image_readme](https://user-images.githubusercontent.com/28053807/180915694-d2f59e7b-5b17-4e3e-8351-f136261fe971.png)

#### - Structure
```
./CNN+LSTM-main
```

#### Requirements
```
+ Python == 3.9.7
+ torch == 1.10.1 
+ scikit-learn >= 1.0.2
+ MATLAB >= MATLAB 9.5 R2018a
+ EEGLAB 2021.1 ver (https://eeglab.org/)
+ FieldTrip Toolbox (www.fieldtriptoolbox.org)
```

#### Usage
+ CNN+LSTM-main : torch code of CNN-LSTM architecture
+ MATLAB        : MATLAB code of dataset load

## DATASET
|Data Name(Release Year)|Resource|Num. of Subject|Device(Num. of Electrode)|Imagery Classes|Cue Display|
|-----------------------|--------|---------------|-------------------------|---------------|-----------|
|Xiaoli(2020)[1]|IEEE DataPort|6|Neuroscan SynAmps2(122)|LH, RH|Arrow|
|Lee(2019)[2]|Deep BCI, MOABB, Gigascience|54|BrainProduct BrainAmp(62)|LH, RH|Arrow|
|Kim(2018)[3]|Deep BCI|12|BrainProduct BrainAmp(30)|LH, RH, RF|Arrow|
|Murat(2018)[4]|Scientific data|13|Neurofax EEG-1200(19)|LH, RH, LF, RF, EF|Object|
|Cho(2017)[5]|Deep BCI, MOABB, Gigascienece|52|Biosemi(64)|LH, RH|Text|
|Shin(2016)[6]|MOABB|29|BrainProduct BrainAmp(30)|LH, RH|Arrow|
|Weibo(2014)[7]|MOABB|10|Neuroscan SynAmps2(64)|LH, RH, F, LHRF, RHLF|Text|
|Ahn(2013)[8]|Deep BCI|10|Biosemi(19)|LH, RH|Arrow|

L: left/ R: right/ H: hand/ F: feet  

### ■ Emotion Attention FER Task

#### It includes scripts for decoding emotion & attention during FER Task through AI models

#### - Structure
```
./DMUE
dataloader.py
deep_face_crop.py
face_recog_crop.py
model.py
train.ipynb
train.py
```

## Acknowledgements
This work was supported by the Republic of Korea's MSIT (Ministry of Science and ICT), under the High-Potential Individuals Global Training Program (No. 2021-0-01537) supervised by the IITP (Institute of Information and Communications Technology Planning & Evaluation).
