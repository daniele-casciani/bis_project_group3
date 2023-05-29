import fnmatch
import os
import json
import numpy as np
import sys
from datetime import datetime
from itertools import compress

import requests.exceptions
import tensorflow
from keras.utils import load_img, img_to_array
from tensorflow import keras
import keras.preprocessing.image
import urllib.request
import keras.applications
from keras.applications.imagenet_utils import preprocess_input

from enum import Enum

from PIL import Image


# cyclone, earthquake, flood, volcano, wildfire
class Disasters(Enum):
    CYCLONE = 0
    EARTHQUAKE = 1
    FLOOD = 2
    VOLCANO = 3
    WILDFIRE = 4


def read_config():
    """ This function reads a JSON configuration file, extracts specific values
        from it (such as input/output directories, timestamps, image directory,
        and placeholder image), and returns them as individual variables.
    """
    f = open('files/configfile.json')
    configs = json.load(f)
    timestamp = datetime.fromisoformat(configs['curr_timestamp'])
    image_dir = configs["image_dir"]
    placeholder_image = configs['placeholder_image']
    output_dir = configs['output_dir']
    return timestamp, image_dir, placeholder_image, output_dir


def update_timestamp(timestamp):
    """ This function updates the 'curr_timestamp' value in the JSON configuration
        file with the provided timestamp and saves the modified file back to disk.
    """
    f = open('files/configfile.json', 'r+')
    content = json.load(f)

    # verify in actual implementation if Z is needed
    content['curr_timestamp'] = timestamp.isoformat()

    f.seek(0)
    str_write = json.dumps(content)
    f.write(str_write)
    f.truncate()
    f.close()


def append_accuracies(images_to_process, accuracies):
    """ this function ensures that the number of images to process matches
        the number of accuracy scores provided, and then it adds each accuracy
        score to the corresponding image dictionary.
    """
    if len(images_to_process) != len(accuracies):
        sys.exit("wrong vectors")
    for i in range(0, len(accuracies)):
        images_to_process[i]["accuracy_score"] = accuracies[i]
    return images_to_process


def write_output(content, filename, directory_l):
    """ this function converts the content object to a JSON string, creates a new
        file in the specified directory with the provided filename, writes the JSON
        string to the file, and then closes the file. The result is a JSON file containing
        the contents of the content object.
    """
    str_write = json.dumps(content)
    newfile = open(os.path.join(directory_l, filename), "w")
    newfile.write(str_write)
    newfile.close()


def incremental_accuracy(old_accuracy, old_n, accuracies, new_n):
    """ this function updates the incremental accuracy by incorporating the previous accuracy
        information (old_accuracy and old_n) with the new accuracy scores (accuracies and new_n).
        The result is the average accuracy across all samples, taking into account both the previous
        and new data.
    """
    cumulative_sum = old_accuracy * old_n
    cumulative_sum = cumulative_sum + np.sum(accuracies)
    return cumulative_sum / new_n


def fetch_image(url, i, placeholder_image, image_dir):
    """ this function downloads an image from a given URL, preprocesses it, and returns the preprocessed
        image for further use in image processing tasks or machine learning models.
    """
    image_name = image_dir + '/image' + str(i)
    try:
        urllib.request.urlretrieve(url, image_name)
    except requests.exceptions.RequestException as e:
        return placeholder_image

    img = load_img(image_name, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img


def empty_image_folder(directory):
    """ this function checks if the specified directory exists and, if it does, it removes all the files
        within that directory. If the directory does not exist, it prints an informative message.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
    else:
        print("folder does not exist")


def ensamble_pred(models, image):
    # Define the weights for each model
    weights = [0.2, 0.3, 0.1, 0.15, 0.25]

    ## Initialize empty list to store predictions from each model
    probabilities = []

    # Do the prediction
    for model in models:
        prediction = model.predict(image)
        probabilities.append(prediction)

    # Calculate weighted scores for each class
    weighted_scores = np.zeros_like(probabilities[0])

    for i, model_pred in enumerate(probabilities):
        weighted_scores += model_pred * weights[i]

    return weighted_scores


def read_inputs(filter_model, classifier, curr_timestamp, image_dir, placeholder_image, event, output_directory):
    """ This is a function that reads inputs from a directory, processes images based on a given filter and classifier,
        and writes the output to another directory. Here's a breakdown of what the function does:
        - Reads the current timestamp from the configuration file
        - Iterates over the files in the input directory
        - Parses each file and inspects the image vector
        - Filters out images using a pre-trained filter model
        - Classifies remaining images using a pre-trained classifier model
        - Calculates the accuracy of each image classification and appends it to the image data
        - Writes the output to a file in the output directory
        - Deletes the processed images from the image directory
        The function also updates the timestamp in the configuration file after all files have been processed.
    """

    # get the time at which we start processing
    # at next iteration, we know that all the images until this time are processed

    # not correct for the demo
    # new_time = datetime.now()

    # iterate on the files
    images = event["images"]
    event_type = event["type"].upper()
    filename = event['event_id']

    # may need to change it in case of ISODate format
    timestamps = np.array([datetime.fromisoformat(obj["date"]) for obj in images])

    # Find the index of the last element with a timestamp lower than the target value
    index = np.searchsorted(timestamps, curr_timestamp, side='right')
    # print(timestamps, curr_timestamp)

    if index > len(images) - 1:
        # update_timestamp(new_time)
        return "", curr_timestamp

    new_time = timestamps.max()

    images_to_process = images[index:]

    # mark for deletion
    to_keep = []
    accuracies = []
    for i in range(0, len(images_to_process)):
        url = images_to_process[i]['image_url']
        image = fetch_image(url, i, placeholder_image, image_dir)
        result = filter_model.predict(image)
        if result[0][0] > result[0][1]:
            to_keep.append(0)
        else:
            # here only relevant image
            result = ensamble_pred(classifier, image)

            # index is the event type since the result is an enum ordered with like the label
            # we just look to the respective index in vector to check only the interested class
            index = getattr(Disasters, event_type).value
            accuracy = result[0][index]
            if accuracy == max(result[0]):
                to_keep.append(1)
                accuracies.append(float(accuracy))
            else:
                to_keep.append(0)
            # cyclone, earthquake, flood, volcano, wildfire

    relevant_images = list(compress(images_to_process, to_keep))
    # if no relevant images for an event, we do not propagate it in out output folder
    # if no new relevant images for an event, we do not update the related file in the output folder
    if len(relevant_images) == 0:
        empty_image_folder(image_dir)
        #update_timestamp(new_time)
        return "", new_time

    done = 0

    images_with_accuracy = append_accuracies(relevant_images, accuracies)

    for filename_out in os.listdir(output_directory):

        # already created a file for this event
        # just need to append the new images
        # calculate new average accuracy incrementally
        if filename_out == filename:
            done = 1
            f_out_temp = os.path.join(output_directory, filename_out)
            f_out = open(f_out_temp)
            out_content = json.load(f_out)
            f_out.close()
            images_out = out_content['images']
            old_accuracy = out_content['average_accuracy']
            old_n = len(images_out)
            new_images = images_out + images_with_accuracy
            out_content['images'] = new_images

            out_content['average_accuracy'] = incremental_accuracy(old_accuracy, old_n, accuracies, len(new_images))
            write_output(out_content, filename_out, output_directory)

            # incremental evaluation of accuracy

    # first images for this event
    # need to create a file for the event and add all the info
    if done == 0:
        event['images'] = images_with_accuracy
        event['average_accuracy'] = np.average(accuracies)
        write_output(event, filename, output_directory)

    empty_image_folder(image_dir)
    #update_timestamp(new_time)

    if done == 1:
        return out_content, new_time
    else:
        return event, new_time


def process_image(event, filter_model, classifier_models):
    """ This is the main function that loads the pre-trained models and calls the read_inputs function.
    """
    curr_timestamp, image_dir, placeholder_image, output_dir = read_config()
    # loop every 15 minutes
    # call if of group 1 for getting directory of data
    output, new_time = read_inputs(filter_model, classifier_models, curr_timestamp, image_dir, placeholder_image, event, output_dir)
    return output, new_time