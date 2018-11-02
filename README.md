# Review Classification
Artificial Neural Network that evaluates if a review is a positive or negative comment.
Python3 Neural Network module.

## Usage
### 1. Clone this module:
```sh
$ git clone https://github.com/rogercoll/ReviewClassification.git
```
### 2. Install libraries:
```sh
$ pip3 install numpy
```
**and**
```sh
$ pip3 install tensorflow
```
### 3. Quick test:
```python
from movierw import NN_Movie_Review

NN = NN_Movie_Review()
#Train value means the maximum value that the words can have in the test and train data

NN.train(10000)
NN.test("excellent film and a brilliant director")
NN.test("Horrible and very bad performance i will never see it again")
```
**Output:**
```
(Loss function, Accuracy)
[0.32592182705879214, 0.87336]
Text: excellent film and a brilliant director
Predictions goes from 0(bad review) to 1(good review)
Your review is a good comment because the final result was:
[0.8149965]
Text: horrible and very bad performance i will never see it again
Predictions goes from 0(bad review) to 1(good review)
Your review is a bad comment because the final result was:
[0.34711733]

```

### 4.More:
You can also change different learning values of the neural network, directly from movierw.py file:

| Property      | Defined Value | 
| ------------- |:-------------:| 
| Vocab_size    |     25000     | 
| Epochs        |       40      |  
| Batch_size    |      512      |  
| Activation    |relu && sigmoid|  
| Optimizer     | AdamOptimizer |  

