# MedNLP
This repository contains code and analysi for ''Disease Identification and NLP answering chatbot for Discharge Summaries'' for MIMIC-III dataset.

The project is aimed at building a system that leverages huge healthcare information available as electronic data and learn a disease identification model.


### Files
* BackEnd.py - A python file containing all the methods containing the structure of the Neural Network model, preprocessing data, ,running the model on the data and finally returning a list of disease found in the given summary.
* Predict.ipynb - A jupyter notebook that prompts users and asks them what summary they would like the model to analyze for them
* Discharge_Summaries.npy - This is a small subset of size 100 of the entire discharge summaries available by the MIMIC-III.
* common_diseases.npy - This is the list of diseases that we used to label the discharge summaries we used as our training data.
* words.npy - This is the list of all words that our system is familiar with.
* model_weights.h10 - This is a file of pretrained weights that will be loaded to the model, with a few adjustment to the BackEnd.py you can retrain the model or let it train for a longer time.

#### Running the Model[Small Subset Data]
We have saved small subset of data to a pickle file **Discharge_Summaries.npy**. 
When user is prompted "Do you want to use one of the pre-existing summaries?[Y,N]" if they select Y, this subset will be loaded and then with the next prompt user can choose which discharge summary from the 100 availble discharge summaries they want the system to analyze. After that the system will take care of preprocessing the data and will feed them to the model and return the list of found diseases.
```python
jupyter notebook
```


#### Running the Model [User Data]
When user is prompted "Do you want to use one of the pre-existing summaries?[Y,N]" if they select N, the user is prompted to type in the summary as a string. User can choose to type in or paste content to the given input box and then After that the system will take care of preprocessing the data and will feed them to the model and return the list of found diseases.
```python
jupyter notebook
```

### Required Dependencies & Libraries
- anaconda 5.2
- Python 3.x
- pandas
- sklearn
- nltk
- numpy
- h5py 2.8.0