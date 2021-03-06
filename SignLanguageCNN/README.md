# Final Project
**PACKAGES/LIBRARIES NEEDED:**
- numpy
- sklearn
- torch

## Train.py
This file comes with 4 defined class models: **Net1, Net2, Net3, Net4.**

**Functions:**
- training_loop: This function trains the model and prints out the associated training loss every 10 epochs. 
- validate: This function is used to validate the training and test accuracy of the 80/20 test sets
- Split_Train_Model: This function splits the training model into 80% training and 20% test. Then it formats those datasets into a list of tuples with the correct dimensions.

### HOW TO USE:
1) Change the code in these lines to the associated names for your files:
	*X_org=np.load('train_data.npy')*
 
	*y_org=np.load('train_labels.npy')*

2) Run the rest of the lines, make sure the model chosen is **Net2()** as that is the final model for the CNN designed.

3) When saving the model make sure to change the **PATH** to the correct directory


## Test.py:
This file has two functions used to test the accuracy of the model with the given data set

**Functions:**
- validate(): This function gets passed in the model as well as the data set that wants to be tested. It prints out the accuracy of the data set being tested. Also prints out the accuracy of each associated class label, and returns the a list of the predicted labels of each tested image.

- Format_data(): This function formats the data to be a list of tuples where each tuple consists of a tensor (the image) and its associated label (an integer in the range 1-9) 

### HOW TO USE:
1) Change the code in these lines to your associated named file:
	*X_org=np.load('input.npy')*

	*y_org=np.load('input.npy')*

2) Make sure the model being loaded is the correct name and in the correct path

3) Run the rest of the code



You may alter this README file.
