from collections import namedtuple
from os import path
from scipy.io import loadmat

"""
This is a supplement to:                                                                             
"Heart-rate Tuned Comb Filters for Processing Photoplethysmogram (PPG) Signals in Pulse Oximetry," a paper by:                                                                                          

Ludvik Alkhoury*, Ji-won Choi*, Chizhong Wang*,                                                      
Arjun Rajasekar†, Sayandeep Acharya†, Sean Mahoney‡,                                                 
Barry S. Shender§, Leonid Hrebien†, and Moshe Kam*                                                   

*Department of Electrical and Computer Engineering                                                   
Newark College of Engineering                                                                        
New Jersey Institute of Technology, New Jersey 07102, USA                                            
Email: {La256@, jc423@, cw278@, kam@}njit.edu                                                        
                                                                                                      
†Department of Electrical and Computer Engineering                                                   
Drexel University, Philadelphia, Pennsylvania 19104, USA                                             
Email: {sa427@, ar924@, lhrebien@ece.}drexel.edu                                                     
                                                                                                      
‡ Regulatory Affairs Department                                                                      
Athena GTX, Johnston, IA 501131, USA                                                                 
Email: smahoney@athenagtx.com                                                                        
                                                                                                       
§Human Systems Department                                                                            
Naval Air Warfare Center Aircraft Division, Patuxent River, MD 20670, USA                            
Email: barry.shender@navy.mil                                                                        
                                                                                                      

Each of the 14 datasets encloses the following:  
1- Time interval of the entire experiment (1st column)  
2- 3-axial X, Y, and Z accelerometer readings (2nd, 3rd, and 4th columns, respectively)  
3- Electrocadiagraphy (ECG) waveform (5th column)  
4- Photoplethysmopraghy (PPG) red and infrared waveforms from channel A (6th column for red and 7th column for infrared, respectively)  
5- Photoplethysmopraghy (PPG) red and infrared waveforms from channel B (8th column for red and 9th column for infrared, respectively)  
6- Nonin 8000R Sensor SpO2 readings (10th column)  

Note: The sampling frequency is 256 Hz

If you use the dataset, please cite the paper as follows: Alkhoury, L., Choi, J. W., Wang, C., Rajasekar, A., Acharya, S., Mahoney, S., ... & Kam, M. (2020). Heart-rate tuned comb filters for processing photoplethysmogram (PPG) signals in pulse oximetry. Journal of clinical monitoring and computing, 1-17.
"""


class SubjectPPGRecord(object):
    def __init__(self, data_file, db_path=".", mat=True):
        self.record_fields = ["time", "acc", "ecg", "ir_a", "red_a", "ir_b", "red_b", "spo2"]
        self.record_idxs = [0, list(range(1, 4)), 4, 5, 6, 7, 8, 9]
        self._fs = 256
        if mat:
            mat_data = loadmat(path.join(db_path, data_file))[data_file]
            self.subject_data = mat_data.T
        else:
            # TODO: loading from text file...
            self.subject_data = []

    @property
    def fs(self):
        return self._fs

    @property
    def raw_data(self):
        return self.subject_data

    @property
    def record(self):
        Signals = namedtuple("Record", self.record_fields)
        return Signals._make([self.subject_data[idx, ...] for idx in self.record_idxs])  # type: ignore
