# Project Name: Event Confirmation with AI
## Description

This GitHub repository contains the code and documentation for Group 3 of the Satellite Data Recommendation System project, developed as part of the BIS 2 course at Politecnico di Milano. The objective of Group 3 is to design and develop an AI module that confirms the type of events by leveraging visual information associated with news articles.

The overall goal of the Satellite Data Recommendation System project, led by eGeos (Leonardo company), is to design a recommendation system that assists in the acquisition of satellite data for monitoring critical events primarily related to security and emergency. Group 3's specific focus is on developing the event confirmation AI module.


## Table of Contents
- [Approach](#approach)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)


## Approach

In addressing the challenge of event confirmation, our focus is on image classification. We employ transfer learning and fine-tuning techniques to leverage pre-trained models for this task.

To train our model, we manually labeled a dataset consisting of 5,000 images per each of the relevant classes: cyclone, earthquake, flood, volcano, and wildfire. This resulted in a total dataset of 25,000 images. We used this dataset to train different kinds of models, which were then combined into an ensemble model. This approach yielded excellent performance in classification for the relevant classes, achieving an average accuracy of 96% over our test set.

During development, we encountered the challenge of discarding irrelevant images that could potentially be fed into the model and misclassified. To mitigate this issue, we trained a binary classifier using 5,000 images for both the relevant and not relevant classes. The not relevant images were obtained from the well-known ImageNet dataset. This binary classifier achieved remarkable results in filtering the two classes, attaining an average accuracy of 99% on the test set.

## Installation

To install and set up the Event Confirmation AI module, follow these steps:

1. Clone this repository to your local machine using the following command:
```shell
git clone https://github.com/daniele-casciani/bis_project_group3
```
2. Navigate to the project directory:
```
cd bis_project_group3
```
3. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

## Usage

To try the Event Confirmation AI module in `sender.py` file there is an example on how is possible to send a request to our server at the provided `url`, it also include an example of the right format of input that would be correctly procesed. 

The module will process the `.json` file including the visual information associated with news articles and provide event type confirmation based on the implemented AI models.

If you want you can also try the code on your localhost by simply substitute the current address with `localhost` as well explained in the code. 

Remember to run the following command:
```
flask --app sample.py run
```

## Features

The Event Confirmation AI module offers the following features:

- Utilizes visual information associated with news articles to confirm the type of events.
- Implements deep learning models to analyze, filter and classify the visual content.
- Provides event type confirmation based on the analysis results.
    
## TODO:
- In sample.py choose the correct mandatory fields
- Find Ram and Storage usage for Paul
