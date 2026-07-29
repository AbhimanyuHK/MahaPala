# MahaPala
Detection of fruits disease by using Machine learning

## Tech Stack

* Python 3.10
* TensorFlow / Keras — CNN model for fruit classification
* OpenCV — template matching sample
* Matplotlib — training/validation charting
* Streamlit — web dashboard
* pytest — automated tests

## Getting Started

Clone the repo and install dependencies:

```
git clone https://github.com/AbhimanyuHK/MahaPala.git
cd MahaPala
pip install -r requirements.txt -r app_requirements.txt
```

For running the test suite, also install the dev dependencies:

```
pip install -r dev-requirements.txt
```

## Usage

Run the dashboard:

```
streamlit run app.py
```

Then open http://localhost:8501 in a browser, upload a fruit/plant image, and view the predicted class and confidence.

Run the tests:

```
pytest test_mahapala.py -v
```

## Architecture 

![image](https://github.com/AbhimanyuHK/MahaPala/assets/32696360/21c38f5d-1875-4dfb-9e92-bfb67f296acd)



## Indentification Of Fruits

Create class to indentify the fruit names tested it for Guava & Mango fruits

* Deep learning algorithm 
* adam optimizer
* relu activation

## Reference

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4245116/


## Model: sequential
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 224, 224, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 224, 224, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2  (None, 112, 112, 16)      0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 112, 112, 32)      4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 56, 56, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 56, 56, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 28, 28, 64)        0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 50176)             0         
                                                                 
 dense (Dense)               (None, 128)               6422656   
                                                                 
 dense_1 (Dense)             (None, 5)                 645       
                                                                 
=================================================================
Total params: 6446885 (24.59 MB)
Trainable params: 6446885 (24.59 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

```
# Dashboard 

_Screenshot below is from the previous Dash-based UI; the dashboard now runs on Streamlit (`streamlit run app.py`)._

![image](https://github.com/AbhimanyuHK/MahaPala/assets/32696360/59d1720c-316d-4f46-a5e8-17e93e6b70aa)

  
## Release Note

## 2026.07.29
* Modernized dependencies (TensorFlow, numpy, matplotlib, opencv-python) and replaced deprecated Keras APIs
* Added pytest-based automated tests and a CI workflow to run them on every push/PR
* Migrated the dashboard from Dash to Streamlit

## 2023.12.16
* #9 Create a sample to identify the fruits name
* #11 Implement visualization dashboard

### 2021.11.01
* #9 Create a sample to identify the fruits name

### 2020.11.01.dev
* Initial setup and environments
* Added a templete matching algorithm 
