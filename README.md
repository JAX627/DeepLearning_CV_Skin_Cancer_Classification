# DeepLearning_CV_Skin_Cancer_Classification
This is my course project of Skin cancer classification, it covers deep learning, computer vision, pytorch.....
Skin Cancer Classification
## Background
The skin cancer classification task is a supervised learning process of machine
learning, recognizing images and correctly classifying the class of skin cancer to the given
labels.
In this skin cancer classification project, it should train and learn from eight types of
labeled skin cancer images through a deep learning computer vision model, and correctly
identify and classify the class of skin cancer in the test set. I need to consider what kind of
the computer vision model, data processing, and the implementation of the training process.
And also adjust the current process based on the feedback from the training results to
achieve better training outcomes. Moreover, according to the observation of image data,
images have different influences like exposure, so I can adopt measures such as image
enhancement, image preprocessing and to improve the robustness and recognition ability of
the model, and also the image numbers of each class is different, I can apply augmentations
to solve the imbalance datasets problem.
## Method
I adopt a pretrained convolutional neural network based on ResNet-50 as the base
model. ResNet-50 is a classic deep residual network that introduces residual connections to
solve the gradient vanishing problem during the training process, and it also have strong
feature extraction capabilities and no need to do extra feature extraction for each images. I
use transfer learning on the base model ResNet-50, fine-tuned the last layer and limited the
output results to 0-8 categories to adapt to the skin cancer classification task.
To train the neural network model, I define the cross-entropy loss function as the loss
function, and used the stochastic gradient descent optimizer with momentum to optimize the
model parameters.
For data preprocessing, I apply different data augmentation operations on the
training set, such as random rotation, random horizontal flip, random vertical flip, color jitter,
Gaussian blur, random sharpening, etc., to increase the diversity of data and improve the
generalization ability of the model. For the validation set, I only performed normalization,
scaling the image pixel values.
## Experiments
### Version 1 - Fine-tune the pre-training base model
I firstly implemented the pre-trained resnet50 model using the PyTorch and
torchvision libraries, and added a fully connected layer to achieve the final output of 8
categories.
At the same time, the data was only normalized, and the folder name was used as
the label for 20 epochs of training.
The training results and test results are as follows:
Based on the training results and test results, I obtain the following information for
optimization:
During the training process, the accuracy is flat around 7-10 epochs, so I can test
different process based on 10 around epochs.
At present, the precision of VCC, BKL, NV, and MEL needs to be improved. The
recall of AK, BKLDF, SCC, and VASC needs to be improved.
### Version 2 - Image Enhancement and Handling Class Imbalance
By analyzing the data, I found: The number of images in the training set is different,
there a few of images for AK, DF, NV, SCC, and VASC. To reduce the impact caused by the
number of training images of each class, I handle class imbalance by adjusting class
weights. Also, noticing that images are affected to varying degrees, I further enhanced the
data.
The training and test results after 20 epochs are as follows:
As shown in the result images, after enhancing the images and handling class
imbalance, the accuracy and various class accuracy have been improved, the accuracy of
the test set has increased from 0.57 to 0.69.
### version 3 - More Image Enhancement + Sampling
This time, I adopt more image enhancements such as color jitter, Gaussian blur, and
random sharpening, and also oversampled the data.
The training and test results after 20 epochs are as follows:
The effect has become worse, and after repeated tests, the more image
enhancements, the worse the effect, the accuracy of the test set has increased from 0.69 to
0.59.
Next, I will continuously adjust image enhancement and handle class imbalance
issues.
### version 4 - Final Version
After repeated attempts, I got the best result:
The test accuracy is 0.715 which is the highest result. This version selectively
retains part of image enhancement and using oversampling to reduce class
imbalance, get the best test set results.

# Appendix
result images
![image](https://github.com/JAX627/DeepLearning_CV_Skin_Cancer_Classification/assets/113168400/4ce6b05d-1854-4f91-a217-2539a87d4270)
![image](https://github.com/JAX627/DeepLearning_CV_Skin_Cancer_Classification/assets/113168400/d837f6be-2185-4942-8e80-cf2c97b2a98d)

training on RTX3090 for half hour for 20 epochs
the weight file in gdrive:
https://drive.google.com/file/d/1sgCkbf38y6j6MpP6haTyHT6rPvFwClKM/view?usp=sharing
