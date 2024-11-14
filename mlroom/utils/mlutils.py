import numpy as np
#import v2realbot.controller.services as cs
import mlroom.utils.ext_services as es
import joblib
from mlroom.config import MODEL_DIR
from datetime import datetime
import socket
import requests
from keras.models import model_from_json
import pickle

def red(text): return f"\033[32m{text}\033[0m"  # Red color
def bold(text): return f"\033[4m{text}\033[0m"  # Bold text

def send_to_telegram(message):
    apiToken = '5836666362:AAGPuzwp03tczMQTwTBiHW6VsZZ-1RCMAEE'
    chatID = '5029424778'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    message = socket.gethostname() + " " + message
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

def get_full_filename(name, version = "1", directory = MODEL_DIR):
  file = name+'_v'+str(version)+'.pkl'
  return directory / file

#LEGACY LOADER
def load_model_legacy(name = None, version = "1", file = None, directory = MODEL_DIR):
  if file is None:
     filename = get_full_filename(name, version, directory)
  else:
     filename = directory / file

  return joblib.load(filename) #somehow support , custom_objects={'SelfAttention': SelfAttention}


#CUSTOM LAYER SUPPORTED SAVE and LOAD- https://chat.openai.com/c/d53c23d0-5029-427d-887f-6de2675c1b1f
#LOADING other model types than keras - https://gemini.google.com/app/5bfbb6cd8c6f39c5
def load_model(name=None, version="1", file=None, directory=MODEL_DIR, cfg_only=False):
    if file is None:
        filename = get_full_filename(name, version, directory)
    else:
        filename = directory / file

    # Load the entire instance with joblib
    loaded_instance = joblib.load(filename)

    # pro cteni metadat, nepotrebujeme cely model jen cfg
    if cfg_only is False:
        if isinstance(loaded_instance.model, dict):  # Check if we deserialized a Keras model 
            model_json = loaded_instance.model['model_json']
            model_weights = loaded_instance.model['model_weights']
            loaded_instance.model = model_from_json(model_json, custom_objects=loaded_instance.custom_layers)
            loaded_instance.model.set_weights(model_weights)

    return loaded_instance

def slice_dict_lists(d, last_item, to_tmstp = False):
  """Slices every list in the dictionary to the last last_item items.
  
  Args:
    d: A dictionary.
    last_item: The number of items to keep at the end of each list.
    to_tmstp: For "time" elements change it to timestamp from datetime if required.

  Returns:
    A new dictionary with the sliced lists.
  """
  sliced_d = {}
  for key in d.keys():
    if key == "time" and to_tmstp:
        sliced_d[key] = [datetime.timestamp(t) for t in d[key][-last_item:]]
    else:
        sliced_d[key] = d[key][-last_item:]
  return sliced_d


#pomocne funkce na manipulaci s daty

def merge_dicts(dict_list):
   # Initialize an empty merged dictionary
    merged_dict = {}

    # Iterate through the dictionaries in the list
    for i,d in enumerate(dict_list):
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
        #vlozime element s idenitfikaci runnera

    return merged_dict

    # # Initialize the merged dictionary with the first dictionary in the list
    # merged_dict = dict_list[0].copy()
    # merged_dict["index"] = []

    # # Iterate through the remaining dictionaries and concatenate their lists
    # for i, d in enumerate(dict_list[1:]):
    #     merged_dict["index"] = 
    #     for key, value in d.items():
    #         if key in merged_dict:
    #             merged_dict[key] += value
    #         else:
    #             merged_dict[key] = value

    # return merged_dict

def convert_lists_to_numpy(data):
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = np.array(value)
    return data

def load_runner(runner_id, data_to_fetch, server, asnumpy = False):
    """
    Vraci pro dany runner data uvedena ve vstupnim listu data_to_fetch
    """
    res, sada = es.get_archived_runner_detail_by_id(runner_id, server)
    if res == 0:
        print("ok")
    else:
        print("error",res,sada)
        raise Exception(f"ERROR loading runner {runner_id} : {res} {sada}")


    ret_dict = {}
    for key in data_to_fetch:
       match key:
          case "bars":
             ret_dict[key] = convert_lists_to_numpy(sada["bars"] ) if asnumpy else sada["bars"] 
          case "indicators":
             ret_dict[key] = convert_lists_to_numpy(sada["indicators"][0]) if asnumpy else sada["indicators"][0]
          case "cbar_indicators":
             ret_dict[key] = convert_lists_to_numpy(sada["indicators"][1]) if asnumpy else sada["indicators"][1]
          case "dailyBars":
             ret_dict[key] = convert_lists_to_numpy(sada["ext_data"]["dailyBars"]) if asnumpy else sada["ext_data"]["dailyBars"]
 
    return ret_dict


def iterate_by_day(concatenated, day_indexes):
    """
    Helper just to show
    how to iterate over concatenated_data with day_indexes
    """
    for day, indexes in enumerate(day_indexes):
        print(f"Day {day + 1}:")
        day_data = {key: {} for key in concatenated}

        for key, (start_index, end_index) in indexes.items():
            print("key:",key)
            # Extract data points for the current day and feature
            for feature, values in concatenated[key].items():
                day_data[key][feature] = values[start_index:end_index]

            print("len keys:", len(list(day_data[key].keys()))," with length ",len(day_data[key]["time"]))
        # Process the day's data as needed
        # For example, print the day's data
        #print(day_data)

#this could concatenate list values of each data, while creating dayily index
#that would allowing to separate daily data if needed
def concatenate_loaded_data(sources):
    """
    Transforms original list structure (days as list)
    [dict(area1=dict(time=[], feature1=[]), area2=dict(time=[], feature1))]

    by creating one flat strucutre and day indexes variable
    dict(area1=dict(time=[], feature1=[]), area2=dict(time=[], feature1))

    day_indexes = {bars: [(0,234),(234,545),(545,88)]}

        [
        {'bars': (0, 4373), 'cbar_indicators': (0, 64559), 'indicators': (0, 4373)},
        {'bars': (4373, 8432), 'cbar_indicators': (64559, 101597), 'indicators': (4373, 8432)}
        ]

    """
    if not sources:
        raise ValueError("The sources list is empty")

    # Dynamically determine the keys from the first day's data
    top_level_keys = sources[0].keys()
    concatenated = {key: {} for key in top_level_keys}
    day_indexes = []

    for day_index, day_data in enumerate(sources):
        day_info = {}
        for key in top_level_keys:
            # Determine the start index
            if day_index == 0:
                start_index = 0
            else:
                # The start index for the current day is the end index of the previous day + 1
                start_index = day_indexes[-1][key][1]

            for feature, values in day_data[key].items():
                # Concatenate feature data
                if feature not in concatenated[key]:
                    concatenated[key][feature] = values
                else:
                    concatenated[key][feature].extend(values)

            # The end index for the current day
            end_index = start_index + len(day_data[key]['time'])
            day_info[key] = (start_index, end_index)
        day_indexes.append(day_info)
    return concatenated, day_indexes
