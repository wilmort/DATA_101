## DATA_101
My final project for the Data 101 Class at the College of Charleston I took in Fall 2020. 

### Overview: 

  For my final project I chose to classify digits using the MNIST Dataset. I chose this dataset because I would like to get familiar working with image data for future projects and also the MNIST set seems to be the print("hello world”) of image recognition. In this project I compare the classification accuracy of two out of the box sci-kit-Learn algorithms, Random Forest and K Neighbors, and a TensorFlow + Keras Convolutional Neural Network. I also take a look at where they fail individually. 

### Setting the stage:

  I start with the standard data science imports of pandas, numpy, pyplot, and seaborn, as well as matplotlib’s image package to help with visualizing the images. 
From sklearn I import the models I want to use, and two metrics, the confusion matrix for checking where the model did well or not so well and accuracy score. I imported TensorFlow and keras for the dataset, convolutional neural network with the Sequential model, 5 different layers, and to_categorical which is similar to sklearns one hot encoder you may be familiar with. 

### Exploratory data analysis:

  I pull the dataset from tf.keras and thankfully, most of the cleaning up is already done and we can get right to it. (To see me cleaning up and wrangling check my Kaggle repo)
    -	I'd like to see what this image data looks like put together.
    -	Because there are 784 pixels in each image, we need to make sure that X train is of shape -1, 28, 28
    - Looks like its in the right shape, so we move along.  
    -	Using matplotlib we can take a peek and see what a sample of the data looks like
    -	If we had not known the context we might not have any idea what this is… a forward slash? Just a random line? So, let’s check out the labels. 
    -	We can do this by printing and sorting the unique values of Y train
    
  Now I’d like to check the distribution
    -	I’m not too worried about exact counts here in this situation, just looking for a fairly even distribution of the values, so I run a quick distplot.
    -	Looks fairly even with the lowest occurrence being 5 slightly below .4 and the highest being roughly .5. 


I think I have enough information now to run some models so let’s get to it. 

### Sk-Learn Models

  Random Forest
    -	First up, I wanted to use a random forest classifier. An issue I ran into was that I kept having to reshape my x’s and y’s throughout this notebook so instead of making a       bunch of transformations globally I moved the reshaping inside of functions for the sklearn models. 
    -	After fitting my X snd Y train and making predations on X test we have a rounded down accuracy of 91%. Not great, not the worst. 
    -	Where does it fail? I couldn’t get seaborn color pallets to bend to my will so, the confusion matrices in this notebook don’t look great. 

  K Neighbor
    -	Next, I wanted to try classifying with a K neighbor algorithm. 
    -	I used the same in function shape as the Random Forest then fit the x and y train and made predictions on X test. Accuracy comes in at 97 (if you’re generous). This is not       great, but again could be worse. 
    -	Where does it fail?


### Convolutional Neural Net
	
  Normalize 
    -	I can normalize the data here by dividing the arrays by 255 giving me arrays with values of 0-1.

  One Hot Encoder
    -	I wanted to create labels with 10 classes of 0-9 so I got some help from Keras to_categorical to do that. 

  Building the CNN
    -	As I mentioned earlier, I am going to use a sequential model to build this network. 
    - The layers I will be passing the data through are Conv2D, MaxPool, Dropout, Flatten and Dense. 

  Hyperparameters
    -	Conv2d Parameters
        Filters - Filters detect spatial patterns such as edges in an image by detecting the changes in intensity values of the image. Choosing 16, 32, and 64 slices the 28x28           images into 49, 24.5, and 12.25 sections. 
        Kernel size - Kernel size is the height and width of the convolution window. I passed it from a small 3x3 to larger 5x5 to 5x5 to a smaller 3x3. 
    -	Max Pooling Parameters - Max pooling is used to reduce the image size by mapping the size of a given window into a single result by taking the maximum value of the                 elements in the window. I chose 2x2 to get 4 quadrants of the images. Stride denotes how many steps we are moving in each step in convolution
    -	Dropout layer: The dropout layer helps to prevent overfitting the model. A drop out of .50 was recommended by Professor Schmidt, I need to look further into how it works. 

  Optimize – Adam optimizer recommended by my Professor, have to look further into optimizers

  Epochs and Batch Size - I set to 300 epochs and 100 batch size.

  Train the model with fit x train and y train
    -	After fitting my X and Y train for 300 epochs and making predations on X test, we have a rounded down accuracy of .996%, pretty good!  

  Where does it fail?


