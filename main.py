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
    filter_dir = configs['filter_dir']
    classifier_dir = configs['classifier_dir']
    timestamp = datetime.fromisoformat(configs['curr_timestamp'])
    image_dir = configs["image_dir"]
    placeholder_image = configs['placeholder_image']
    output_dir = configs['output_dir']
    return timestamp, image_dir, placeholder_image, filter_dir, classifier_dir, output_dir


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

    new_time = datetime.now()
    # iterate on the files
    images = event["images"]
    event_type = event["type"].upper()
    filename = event['id']

    # may need to change it in case of ISODate format
    timestamps = np.array([datetime.fromisoformat(obj["date"]) for obj in images])
    # Find the index of the last element with a timestamp lower than the target value
    index = np.searchsorted(timestamps, curr_timestamp, side='right')
    # print(timestamps, curr_timestamp)

    if index > len(images) - 1:
        update_timestamp(new_time)
        return 0

    images_to_process = images[index:]

    # mark for deletion
    to_keep = []
    accuracies = []
    for i in range(0, len(images_to_process)):
        url = images_to_process[i]['URLImage']
        image = fetch_image(url, i, placeholder_image, image_dir)
        result = filter_model.predict(image)
        if result[0][0] > result[0][1]:
            to_keep.append(0)
        else:
            result = classifier.predict(image)
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
        update_timestamp(new_time)
        return 0

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
    update_timestamp(new_time)


def main():
    """ This is the main function that loads the pre-trained models and calls the read_inputs function.
    """
    curr_timestamp, image_dir, placeholder_image, filter_dir, classifier_dir, output_dir = read_config()
    event = {"id": "gr1-Turkey-2023_02_06T00_00_00-2023_02_06T12_00_00-earthquake", "country": "Turkey", "date": "2023-02-06 01:17:34", "type": "wildfire", "timeframe": ["2023-02-06 00:00:00", "2023-02-06 12:00:00"], "locations": [[37.0143, 37.2256], [37.0001, 37.2252], [36.8929, 37.1893], [37.9141, 37.7712], [38.0613, 37.9227], [36.9658, 37.1816], [38.5342, 38.1845], [37.1962, 38.0106], [38.0984, 38.0315], [37.2033, 37.9962], [37.8023, 38.0249], [38.1847, 38.248]], "images": [{"URLTweet": "https://twitter.com/GeorgeMelikyan/status/1622416823613276162", "URLImage": "data:image/png;base64,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", "date": "2023-02-06 02:08:08"}]}
    filter_model = keras.models.load_model(filter_dir)
    classifier = keras.models.load_model(classifier_dir)

    # loop every 15 minutes
    # call if of group 1 for getting directory of data
    read_inputs(filter_model, classifier, curr_timestamp, image_dir, placeholder_image, event, output_dir)


main()
