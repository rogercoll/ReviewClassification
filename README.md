# ReviewClassification
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
```
$ pip3 install tensorflow
```
### 3. Use it in your code:
```python
from movierw import NN_Movie_Review

NN = NN_Movie_Review()
#Train value means the maximum value that the words can have in the test and train data

NN.train(10000)
NN.test("excellent film and a brilliant director")
NN.test("Horrible and very bad performance i will never see it again")
```
