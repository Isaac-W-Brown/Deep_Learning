# Deep_Learning
Deep learning for computer vision of numbers and letters


## EMNIST database
The EMNIST database of numbers can be downloaded at https://www.kaggle.com/crawford/emnist. Download emnist-balanced-test.csv and emnist-balanced-train.csv - these have equal numbers of each class of character. Save these into the same folder and note the file path.

## Set Up 
* Download the repository
* Run `requirements.txt`
* In `EMNIST_Data.py`, change the file directory to where you saved the EMNIST data CSVs.
  ```
  import os
  os.chdir("c://Users/... {YOUR FILE PATH HERE} ")
  ```
* Run `EMNIST_Data.py`. Check that `EMNIST Training Data.file` and `EMNIST Training Data.file` have both been saved to the same file directory as the EMNIST CSVs.
* In `Convolutional.py` and `FullyConnected.py` change the file directory to be the same as in `EMNIST_Data.py`

`Convolutional.py` and `FullyConnected.py` are now ready to run.

`Speed_Test.py` is used to compare the single threaded CPU performance of computers at matrix multiplication
