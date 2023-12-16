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
def load_model(name = None, version = "1", file = None, directory = MODEL_DIR):
  if file is None:
     filename = get_full_filename(name, version, directory)
  else:
     filename = directory / file

  # Load the entire instance with joblib
  loaded_instance = joblib.load(filename)

  # Deserialize the Keras model
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

def load_runner(runner_id, use_cbars = False):
    res, sada = es.get_archived_runner_detail_by_id(runner_id)
    if res == 0:
        print("ok")
    else:
        print("error",res,sada)
        raise Exception(f"ERROR loading runner {runner_id} : {res} {sada}")

    #print(sada)
    bars = sada["bars"]
    #vracime tick indicatory if required
    if use_cbars:
      indicators = sada["indicators"][1]
    else:
      indicators = sada["indicators"][0]
    return bars, indicators
