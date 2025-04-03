
# Assignment 2 (due Apr 18th)

This assignment covers convolutional neural networks (CNNs) and recurrent neural networks (RNNs), specifically LSTMs. You will be working in teams of of 1 or 2 students to complete the tasks below.  Feel free to use any libraries or frameworks you are comfortable with, but we recommend using [PyTorch](https://pytorch.org/) or [TensorFlow/Keras](https://www.tensorflow.org/) for this assignment, as they provide high-level APIs for building and training deep learning models.

## Setup

We strongly recommend using the [Unity](unity.uri.edu) resources for this assignment.  The details of setting up accounts were provided in class.  If you have not set up your account, please do so before starting the assignment.

> [!IMPORTANT]
> The datasets for this assignment will be under a shared folder in Unity.  You don't want to download the datasets from other sources, as the data we are sharing are formatted specifically for this assignment.

## Task 1: Image Classification

In this task, you are given a dataset of images and their corresponding labels (within the folder structure).  Your goal is to train a convolutional neural network (CNN) to classify the images into their corresponding labels.

The dataset is available under the shared folder: `/work/pi_csc561_uri_edu/datasets/intel-images` and corresponds to the *Intel Image Classification* competition, initially published on [Analytics Vidhya](https://datahack.analyticsvidhya.com/).  The dataset contains 150x150 images of 6 different classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.

The dataset is split into two folders:

- `training`: 14034 images, organized into subfolders by their corresponding labels. Each subfolder corresponds to a class, and contains images belonging to that class;
- `test`: 3000 images -- no labels available.

For this task, you can design your own CNN architecture, or you can use a pre-trained model (e.g., ResNet, VGG, etc.) and fine-tune it on the dataset.  Eventually, you can also test if data augmentation improves your model's performance.

The test set will be used to evaluate your final model.  As no explicit validation set is provided, you will need to split the training set into a training and validation.  We recommend using 80% of the training set for training and 20% for validation.  You can use any method you like to split the dataset, but we recommend using stratified sampling to ensure that the classes are evenly distributed in both sets.

Once you are done with training and validation, you will get the predictions of your model using the test set.  Save the predictions into `cnn-predictions.txt`.  The format of the text file should be as follows:

```
image_id_1 label_1
image_id_2 label_2
...
image_id_n label_n
```

where `image_id_i` is the file name of the image and `label_i` is the predicted label.  Make sure the file names of the images do not include any folders or `'/'` characters.  Each label should be an integer value corresponding to the class of the image, as follows:

```
buildings: 0
forest: 1
glacier: 2
mountain: 3
sea: 4
street: 5
```

### Submission
You will submit the following files via Gradescope:

- `cnn-predictions.txt`: the predicted labels for the test set (it will be used for grading and generating an entry for the leaderboard)
- **other source files**: all source files used to train, validate, and test your models.  Note that we only require you to submit source files, these include `.py`, `.ipynb`, and `.sh` files.  No other files are required.  Please avoid naming files with whitespaces or special characters other than `'-'`.

## Task 2: Image Description Generation

In this task, you will be given a dataset of images and their corresponding descriptions in English.  Your goal is to train a language model using LSTMs, conditioned on the image features, to generate a description for a given image.

The data is available in a shared folder under `/work/pi_csc561_uri_edu/datasets/multi-30k` and corresponds to the *Multi30K: Multilingual English-German Image Descriptions* dataset.  The dataset contains images and their corresponding descriptions in English and German.  You will only use the English descriptions for this task.  The dataset is split into three parts:

- `training`: 29000 images, each with 5 English descriptions
- `validation`: 1014 images, each with 5 English descriptions
- `test`: 1000 images -- no descriptions available

For this task, you will need to extract features from the images using a pre-trained model.  Feel free to use any pre-trained model, from your favorite CNN, to a more sophisticated model like [CLIP](https://github.com/openai/CLIP).  The quality of the features will have an impact on the performance of your model, so choose wisely.  You will need to extract features for all the images in the training, validation, and test sets.  The **features should be extracted and saved once** in a format that can be easily loaded into your model (see `torch.save` and `torch.load` in PyTorch). This will save you time during training, as you won't need to re-extract features every time you run your model.

You will be using the training set to train your model and the development set to perform model selection (hyperparameter selection).  We strongly recommend that during the initial coding stage, you use a very small subset of the training set.  Once you have a working code that **successfully** trains a model, you can train using the full training set.  

The test set will be used to evaluate your final model.  Once you are done with training and validation, pass the test set of images to your model and generate the descriptions.  Save the generated descriptions into `lm-predictions.txt`.  The format of the text file should be as follows:

```
image_id_1: description_1
image_id_2: description_2
...
image_id_n: description_n
```

where `image_id_i` is the id of the image and `description_i` is the generated description.  The image ids are the file names of the images in the test set.  Make sure the file names of the images do not include any folders or `'/'` characters.

### Submission

You will submit the following files via Gradescope:

- `lm-predictions.txt`: the generated descriptions for the test set (it will be used for grading and generating a leaderboard)
- **other source files**: all source files used to train, validate, and test your models.  Note that we only require you to submit source files, these include `.py`, `.ipynb`, and `.sh` files.  No other files are required.  Please avoid naming files with whitespaces or special characters. 
