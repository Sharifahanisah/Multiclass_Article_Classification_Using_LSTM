# Multiclass_Article_Classification_Using_LSTM
 <3
 :wave:
 :raised_hand_with_fingers_splayed:
 [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

### The Project's Aim 
To devalope a LSTM model to categorize unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics 
 
## Data Source:
https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

## Tensorboard Plot

![Tensorboard](https://user-images.githubusercontent.com/109563861/180977791-364b0b80-69b9-4083-8eb3-12c69be2f8f1.PNG)

* BLUE == Train
* PINK == Validation

From the Tensorboard graph:
>>    * From the graph it can be seen that overfit occure 
>>    * As performance on the train set is good and continues to improve, whereas performance on the validation set improves to a point and then begins to degrade.
>>    *  As the train loss slopes down and the validation loss slopes down, hits an inflection point, and starts to slope up again.


## Performance of the model F1 score

![F1_score](https://user-images.githubusercontent.com/109563861/180978151-42d571e1-1996-4191-ae32-4c59d1189c4d.PNG)

From the model F1:
>>    * The F1 Score is 0.93

## Performance of the model accuracy

![accuracy](https://user-images.githubusercontent.com/109563861/180978226-0d742a03-acdf-4e1f-b1b6-1538c3224132.PNG)

From the model accuracy:
>>    * The accuracy is 92%

## Architecture of the Model

![model](https://user-images.githubusercontent.com/109563861/180978318-dcc73567-9b41-4d1a-ac9c-05390ecfb120.png)
