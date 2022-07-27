# Extraction Biomarker of Motor Imagery Task
>This is the AI-based Brain Signal Processing Algorithm - Motor Function project. They consist of EEG signal classification and dataset loading code. It is the Convolutional Neural Network (CNN)-Long Short-Term Memory (LSTM) models for signal processing and classification on motor imagery, written in PyTorch and Tensorflow.  It also contains code for loading multiple datasets written in MATLAB.

This is a schematic diagram of the proposed CNN-LSTM model. We propose a new model by referring to the following paper: Zhang, Ruilong, et al. "Hybrid deep neural network using transfer learning for EEG motor imagery decoding." Biomedical Signal Processing and Control 63 (2021): 102144.
![image_readme](https://user-images.githubusercontent.com/28053807/180915694-d2f59e7b-5b17-4e3e-8351-f136261fe971.png)

## Requirements
+ Python == 3.9.7
+ torch == 1.10.1 
+ scikit-learn >= 1.0.2
+ MATLAB >= MATLAB 9.5 R2018a
+ EEGLAB 2021.1 ver (https://eeglab.org/)
+ FieldTrip Toolbox (www.fieldtriptoolbox.org)

## Usage
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

[1]	X. Wu, “Ear-EEG Recording for Brain Computer Interface of Motor Task.” IEEE, Feb. 18, 2020. Accessed: Jul. 13, 2022. [Online]. Available: https://ieee-dataport.org/open-access/ear-eeg-recording-brain-computer-interface-motor-task  
[2]	M.-H. Lee et al., “EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy,” GigaScience, vol. 8, no. giz002, May 2019, doi: 10.1093/gigascience/giz002.  
[3]	K.-T. Kim, H.-I. Suk, and S.-W. Lee, “Commanding a Brain-Controlled Wheelchair Using Steady-State Somatosensory Evoked Potentials,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 26, no. 3, pp. 654–665, Mar. 2018, doi: 10.1109/TNSRE.2016.2597854.  
[4]	M. Kaya, M. K. Binli, E. Ozbay, H. Yanar, and Y. Mishchenko, “A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces,” Sci Data, vol. 5, no. 1, Art. no. 1, Oct. 2018, doi: 10.1038/sdata.2018.211.  
[5]	H. Cho, M. Ahn, S. Ahn, M. Kwon, and S. C. Jun, “EEG datasets for motor imagery brain–computer interface,” Gigascience, vol. 6, no. 7, Jul. 2017, doi: 10.1093/gigascience/gix034.  
[6]	J. Shin et al., “Open Access Dataset for EEG+NIRS Single-Trial Classification,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 10, pp. 1735–1745, Oct. 2017, doi: 10.1109/TNSRE.2016.2628057.  
[7]	W. Yi et al., “Evaluation of EEG Oscillatory Patterns and Cognitive Process during Simple and Compound Limb Motor Imagery,” PLOS ONE, vol. 9, no. 12, p. e114853, 9 2014, doi: 10.1371/journal.pone.0114853.  
[8]	M. Ahn et al., “Gamma band activity associated with BCI performance: simultaneous MEG/EEG study,” Frontiers in Human Neuroscience, vol. 7, 2013, Accessed: Jul. 13, 2022. [Online]. Available: https://www.frontiersin.org/articles/10.3389/fnhum.2013.00848
