# Diabetic Retinopathy Detection

## Introduction
In the healthcare field, the treatment of diseases is more effective when detected at an early stage. Diabetic Retinopathy (DR) is a complication of diabetes that causes the blood vessels of the retina to swell and to leak fluids and blood. DR can lead to a loss of vision if it is in an advanced stage. Worldwide, DR causes 2.6% of blindness.

The automated methods for DR detection are cost and time saving and are more efficient than a manual diagnosis. A manual diagnosis is prone to misdiagnosis and requires more effort than automatic methods. A trained doctor can only detect with 50% accuracy.
Retina dataset
The dataset given to us had 3662 fundus images that were resized to 224x224 and gaussian filtered. Images belonged to 5 classes namely - <br>
0 - Negative or No DR: Patient has no disease. <br>
1 - Mild DR (Stage 1): Patient has mild level of disease. <br>
2 - Moderate DR (Stage 2): Patient has moderate level of disease. <br>
3 - Severe DR (Stage 3): Patient has severe level of disease, the most part of the retina is damaged, can lead to complete blindness. <br>
4 - Proliferative DR (Stage 4): Patient has proliferative levels of disease. The patient’s eye is damaged to an extent where treatment is elusive, about 80 percent of blindness exists. <br>
## Image preprocessing
Preprocessing was required to balance the dataset and extract important features. Noise was already filtered by the gaussian filter. Data augmentation techniques were performed when some image classes were imbalanced and to increase the dataset size. Data augmentation techniques include translation, rotation, shearing, flipping and contrast scaling. 

## Model Architecture
Dataset was splitted into training and validation set with an 80:20 split. At first I used the conventional CNN’s to extract features from the fundus image. The accuracy was really bad. CNN’s are faster than NN’s and are able to capture features with respect to surrounding pixels. So I switched to more complex models that have proved their gravity.
### Transfer Learning
In this I used the architectures that were predefined and removed the top layer to add my own ‘softmax’ layer that can classify between 5 classes of DR. I tried with VGG19 initially and that gave me an accuracy of 65%. ResNet50 gave a boost in accuracy as it is a 152 layer deep network with skip-connections.
Dataset was highly imbalanced which was tackled to some extent by data augmentation. Another way of doing it was class weights. Classes that had a very few number of samples were given higher weightage. This ensured that the model was not biased to a single class with largest samples (No DR in our case). 
The veins in the image are used for detecting DR. They appeared to be hazy and unclear. I used Contrast Limited Adaptive Histogram Equalisation(CLAHE) to improve the contrast in the image. This helps to get a better view of veins in the retina.
Before and After applying CLAHE : 

After giving each class a weight and applying CLAHE, VGG16 architecture was used to train the model. Last 16 layers of the VGG model were set trainable to get better results as the pretrained weights are based on ‘imagenet challenge’ which had very different kinds of images. A softmax layer with 5 output probabilities was added to ‘fc2’. It takes an input image of size (#,224,224,3) and outputs probabilities related to each class of Diabetic Retinopathy.
Early Stopping was used to monitor validation accuracy and stop after a patience of 10 epochs. Model was set to train for 80 epochs but stopped at 30 due to no change in validation accuracy. If I trained the model further, overfitting could have occurred. 
Batch size of 32 was used. ImageDataGenerator was used to fetch images from the folder.
Hyperparameter optimization
Grid search was used to search for the best set of hyperparameters. Adam optimizer with a learning rate of 0.000001 was found to perform best in terms of accuracy.


## Model evaluation
Model took around 15 minutes to train on google colab with GPU and 12GB of RAM. Model started with an accuracy of 50% and reached 98% by 30 epochs. 
Model accuracy Vs #epochs

## Model loss Vs epochs

The metrics used for measuring the performance of the model were - 


## Accuracy
98% on training set and 81% on validation set
F1-score
0.65
Precision
0.65
Recall
0.64


## Conclusion
Using this model we got promising results with a comparatively small dataset. Accuracy was better than a trained doctor. The model was tested on unseen data to test the generalizability of the model. Our results show that the model has been able to achieve good performance due to the various techniques that we used at every stage of the ML pipeline. This model was deployed using Flask on a local machine.
