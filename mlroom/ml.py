import os
os.environ["KERAS_BACKEND"] = "jax"
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.models import Sequential
from mlroom.utils.enums import PredOutput, Source, TargetTRFM, ScalingMode
from mlroom.config import DATA_DIR, SOURCES_GRANULARITY
import joblib
from mlroom.utils import mlutils as mu
from mlroom.utils.mlutils import red, bold
#import utils.mlutils as mu
#from .utils import slice_dict_lists
import numpy as np
from copy import deepcopy
#import v2realbot.controller.services as cs
from mlroom.utils import ext_services as exts
import mlroom.arch as arch
import requests
import keras.callbacks as cb
from keras.models import model_from_json, load_model, Model
import inspect
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time
from traceback import format_exc
from sklearn.utils.class_weight import compute_class_weight
#Basic classes for machine learning
#drzi model a jeho zakladni nastaveni

#Sample Data
sample_bars = {
    'time': [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15],
    'high': [10, 11, 12, 13, 14,10, 11, 12, 13, 14,10, 11, 12, 13, 14],
    'low': [8, 9, 7, 6, 8,8, 9, 7, 6, 8,8, 9, 7, 6, 8],
    'volume': [1000, 1200, 900, 1100, 1300,1000, 1200, 900, 1100, 1300,1000, 1200, 900, 1100, 1300],
    'close': [9, 10, 11, 12, 13,9, 10, 11, 12, 13,9, 10, 11, 12, 13],
    'open': [9, 10, 8, 8, 8,9, 10, 8, 8, 8,9, 10, 8, 8, 8],
    'resolution': [1, 1, 1, 1, 1,1, 1, 1, 1, 1,1, 1, 1, 1, 1]
}

sample_indicators = {
    'time': [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15],
    'fastslope': [90, 95, 100, 110, 115,90, 95, 100, 110, 115,90, 95, 100, 110, 115],
    'fsdelta': [90, 95, 100, 110, 115,90, 95, 100, 110, 115,90, 95, 100, 110, 115],
    'fastslope2': [90, 95, 100, 110, 115,90, 95, 100, 110, 115,90, 95, 100, 110, 115],
    'ema': [1000, 1200, 900, 1100, 1300,1000, 1200, 900, 1100, 1300,1000, 1200, 900, 1100, 1300]
}

#Trida, která drzi instanci ML modelu a jeho konfigurace
#take se pouziva jako nastroj na pripravu dat pro train a predikci
#pozor samotna data trida neobsahuje, jen konfiguraci a pak samotny model
"""
Zatim bud bud 1) bar nebo stanndardni indicator 2) cbar indikatory  , zatim nepodporovano spolecne protoze nemaji stejny time
"""
class ModelML:
    # Registry for custom layers
    custom_layers = {}
    #pri inciializaci modelu sem ulozime Custom Layers architectury, ktere se pak naloaduji pri importu
    # custom_layers = {
    #     'CustomLayer1': CustomLayer1,
    #     'CustomLayer2': CustomLayer2
    # }
    def __init__(self,
                cfg: dict = None, #whole parsed self.cfguration
                cfg_toml: str = None, #whole configuration in unparsed toml for later use
                pred_output: PredOutput = PredOutput.LINEAR, #not used DECOMM
                train_target_steps: int = 0, #train not used DECOMM
                train_target_transformation: TargetTRFM = TargetTRFM.KEEPVAL, #train not used DECOMM
                )-> None:

        def sort_input_lists(input_dict):
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    sort_input_lists(value)
                elif isinstance(value, list):
                    value.sort()

        self.name = cfg.get("model",{}).get("name", None)
        self.version = cfg.get("model",{}).get("version", 0.1)
        self.note  = cfg.get("model",{}).get("note", None)
        self.architecture = cfg.get("model",{}).get("architecture", None)
        self.input = cfg.get("model",{}).get("input", None)
        self.target = cfg.get("model",{}).get("target", None)

        self.cfg = cfg
        #pro zpetne dohledani
        self.metadata = dict(cfg=cfg, cfg_toml=cfg_toml, history={})
        self.pred_output: PredOutput = pred_output
        self.model = Sequential()

        #sortneme vsechny listy
        sort_input_lists(self.input)

        # Extracting distinct values of the sources and initi the scalers 
        # Initializing scalersX (now separate scaler for each input. Type of scaler defined here (standard/minMax)
        # TODO: separate scaler for feature
        self.distinct_sources = set()
        self.scalersX = {}
        #INIT scalersX for each input
        for key, value in self.input.items():
            for k in value:
                if value.get("scaler", False):
                    # Dynamically getting the scaler class based on the name
                    # Initializing the scaler
                    ScalerClass = getattr(preprocessing, value["scaler"])
                    self.scalersX[key] = ScalerClass()
                #add distinct sources
                if k in list(SOURCES_GRANULARITY.keys()):
                    self.distinct_sources.add(k)

        #pokud nemam v distinct sources areu targetu, tak pridam (pro situace, kdy napr. bar target a jen cbar vstup)
        target_area = list(self.target.keys())[0]
        if target_area not in self.distinct_sources:
            self.distinct_sources.add(target_area)
        
        #INIT scalerY (scaler from target settings)
        self.scalerY = None
        if self.target.get("scaler", False):
            ScalerClass = getattr(preprocessing, self.target.get("scaler", "StandardScaler"))
            self.scalerY = ScalerClass()

        #Extract all features grouped by distinc_sources (ie. bars = time, close cbars_indicators = time, close -..)
        #needed for features validation
        self.features_required= {}
        for source in self.distinct_sources:
            unique_values = set()
            for key in self.input:
                if source in self.input[key]:
                    unique_values.update(self.input[key][source])
            self.features_required[source] = unique_values

        self.highest_res_key = self.get_highest_resolution()

        train_runner_ids = self.cfg.get("train",{}).get("runners", None)
        train_batch_id = self.cfg.get("train",{}).get("batch", None)
        train_remove_cross_sequences = self.cfg.get("train",{}).get("remove_cross_sequences", True)
        train_runners_per_batch = self.cfg.get("train",{}).get("runners_per_batch", None)
        
        if (train_runner_ids is None or len(train_runner_ids) == 0) and train_batch_id is None:
            raise Exception("train_runner_ids nebo train_batch_id musi byt vyplnene")
        self.train_runner_ids = train_runner_ids
        self.train_batch_id = train_batch_id
        self.train_target_steps = train_target_steps
        self.train_runners_per_batch = train_runners_per_batch
        self.train_target_transformation = train_target_transformation
        #keep cross sequences between runners
        self.train_remove_cross_sequences = train_remove_cross_sequences
        self.custom_layers = {}

    #INIITIALIZE and COMPILE MODEL ARCHITECTURE
    def initialize_model(self, X_train):
            print("Batch 1 - INITIALIZING MODEL")
            input_shape = []
            for idx, val in enumerate(X_train):
                input_shape.append((val.shape[1], val.shape[2]))

            #LOAD and INITIALIZE THE ARCHITECTURE
            model_name = self.architecture["name"]
            model_inputs = len(self.cfg['model']['input'])
            if model_inputs==0:
                print("model inputs not defined in cfg")
                return

            if len(X_train) != model_inputs:
                print(f"Number of inputs doesnt match model requirements len(X_train){len(X_train)} vs model input cfg {model_inputs}")
                return

            model_params = self.architecture.get("params", {})

            print(red(f"MODEL: {model_name}"))
            print("MODEL PARAMS:", red(model_params))

            arch_function = eval("arch."+model_name+"."+model_name)
            print("FUNCTION TO CALL",arch_function)

            self.metadata["arch_function"] =  inspect.getsource(arch_function)
            #print("INSPECTING THE ARCH FUNC",self.metadata["arch_function"])

            # **model_params
            self.model, self.custom_layers = arch_function(input_shape, **model_params)
            print("COMPILED MODEL LOADED")

    
    def load_validation_data(self):
        """Loads external validation data from runners or batch 
            returns tuple (X,Y) or None
        """
        validation_runners = self.cfg.get("validation",{}).get("runners", None)
        validation_batch = self.cfg.get("validation",{}).get("batch", None)
        if (validation_runners is not None and len(validation_runners) > 0) or validation_batch is not None:
            print(red(f"Loading validations {validation_runners=} {validation_batch=}"))
            source_data = self.load_data(runners_ids=validation_runners, batch_id=validation_batch)
            concatenated, day_indexes = mu.concatenate_loaded_data(source_data)
            #for validation data - just transform, no fit
            X_test, y_test = self.scale_and_sequence(concatenated, day_indexes, scaling=ScalingMode.ONLY_TRANSFORM)
            X_test = list(X_test.values())
            print(red("VALIDATION DATA SEQUENCED"))
            return (X_test, y_test)
        else:
            print("No validation runners or batch provided")
            return None

    def load_best_model(self, res_object, key):
        """
        Loads back model with best metrics in ModelCheckpoint
        """
        fit_cfg = self.cfg["fit"]
        if self.cfg.get("save_best", False):
            try:
                filepath = fit_cfg["callbacks"]["ModelCheckpoint"]["filepath"]
                #monitored metrics and its best value and epoch
                metric = fit_cfg["callbacks"]["ModelCheckpoint"]["monitor"]
                monitor_dict = res_object.history[metric]
                best_epoch = monitor_dict.index(min(monitor_dict)) + 1 
                best_value = min(monitor_dict)
                if self.metadata["history"].get("saved_best", None) is None:
                    self.metadata["history"]["saved_best"] = {}
                self.metadata["history"]["saved_best"][key] = {"epoch": best_epoch, metric: best_value}
                self.model = load_model(filepath)
                print(red(f"CHECKPOINTED MODEL USED [{metric}]: {best_value} epoch: {best_epoch}"))
            except Exception as e:
                print(red("ERROR in CHECKPOINT"+ str(e)+ format_exc()))

    def populate_fit_params(self, validation_tuple = None, y_train = None):
       #POPULATE FIT PARAMS (key callback is processed)
        # we use all excepts callbacks
        fit_cfg = self.cfg["fit"]
        fit_params = {k: v for k, v in fit_cfg.items() if k != "callbacks" or k!="class_weight"}

        #INIT CALLBACKS from PARAMS
        if fit_cfg.get("callbacks", False):
            fit_params['callbacks'] = []
            for callback_name, callback_params in fit_cfg["callbacks"].items():
                CallbackClass = getattr(cb, callback_name)
                fit_params['callbacks'].append(CallbackClass(**callback_params))

        #load external validation data during fit, if required
        if self.cfg.get("validation",{}).get("validate_during_fit", False) is True and validation_tuple is not None:
            fit_params["validation_data"] = validation_tuple
        
        #CLASS WEIGHT
        if self.cfg["fit"].get("class_weight", False):
            class_weight = self.cfg["fit"]["class_weight"]
            #jde primo o dictionary, davame hodnoty
            if isinstance(class_weight, dict):
                # Convert string keys to integer keys (bc of TOML)
                class_weight_int_keys = {int(key): value for key, value in class_weight.items()}
                fit_params["class_weight"] = class_weight_int_keys
                print(bold(f"Class_weight set to {class_weight_int_keys}"))
            #jde o string ("balanced") vypocitavama distribuci automaticky
            elif isinstance(class_weight, str):
                y_train_1d = y_train.ravel() 
                #Calculate class weights distribution calculation
                class_weights = compute_class_weight(class_weight, classes=np.unique(y_train_1d), y=y_train_1d)
                class_weight_dict = dict(enumerate(class_weights))
                print(bold(f"Calculated class_weight {class_weight_dict}"))
                fit_params["class_weight"] = class_weight_dict
            else:
                print("not valid setting in class_weight")

        return fit_params

    #X_train bude list
    def train(self, X_train, y_train, batch_number = 1, total_batches = 1, validation_tuple = None):
        #populate params that go to model.fit
        fit_params = self.populate_fit_params(validation_tuple, y_train)

        #INIT MODEL on FIRST BATCH
        if batch_number == 1:
            self.initialize_model(X_train)
        else:
            print(bold("RETRAINING THE MODEL"))

        res_object=self.model.fit(X_train,y_train, **fit_params)
        key = f"batch_{batch_number}_{total_batches}"
        self.metadata["history"][key]=res_object.history

        #if it was required, loads back model with best metrics 
        self.load_best_model(res_object, key)

        mu.send_to_telegram("TRAINING FINISHED")

    #TRAIN and SAVE/UPLOAD - train the model and save or upload it according to cfg
    def train_and_store(self, X_train, y_train, batch_number, total_batches, validation_tuple = None):

        self.train(X_train, y_train, batch_number, total_batches, validation_tuple)
        #save the model
        self.save()

        if self.cfg["upload"]:
            res, val = self.upload()
            if res < 0:
                print("ERROR UPLOADING",res, val)
                return
            else:
                print("Model UPLOADED", res)

    #PUVDNO SAVE bez podpory custom layers
    def save_legacy(self):
        filename = mu.get_full_filename(self.name,self.version)
        joblib.dump(self, filename)
        print(f"model {self.name} save")

    #CUSTOM LAYER SUPPORTED SAVE and LOAD- https://chat.openai.com/c/d53c23d0-5029-427d-887f-6de2675c1b1f
    def save(self):
        filename = mu.get_full_filename(self.name,self.version)

        # Branch based on model type
        if isinstance(self.model, Model):  # Check if it's a Keras model
            # Save the Keras model separately
            model_json = self.model.to_json()
            model_weights = self.model.get_weights()
            # Replace the model attribute with its serialized form
            self.model = {'model_json': model_json, 'model_weights': model_weights}

        # Use joblib to serialize the entire instance (other types that Keras)
        joblib.dump(self, filename)

        # Restore the model attribute to its original state (Keras only)
        if isinstance(self.model, dict): 
            self.model = model_from_json(model_json, custom_objects=self.custom_layers)
            self.model.set_weights(model_weights)

    def upload(self):
        filename = mu.get_full_filename(self.name,self.version)
        return exts.upload_file(filename, self.cfg['upload_server'])

    #create X data with features
    #Pro každý typ vstupu vytvorime samostatne pole vstupe dle danych indikatoru
    #vstup sources obsahuje dictionary se sources
    #TOTO NEJAK VYMYSLET, ABY TO KOPIROVALO LOGIKU TRAININGU
    #TAKY VYMYSLET ABY SE TOTO NEMUSELO VOLAT PRI KAZDEM ITERACI v RT- zychlit
    #ORIZNEME SEKVENCI PRIMO ZDE
    #TODO TOTO konzolidovat


    #toto bude vlastne sekvencing pro skalarni predict
   

    #TODO - filter settings on source
    # a = {
    #     'cbar_indicators': ['tick_price', 'tick_trades', 'tick_volume'],
    #     'bars': ['feat1']
    # }

    # b = {
    #     'bars': {'feat1': [1, 2, 3, 4], 'FEAT2': []},
    #     'cbar_indicators': {'tick_price': [], 'tick_trades': [], 'tick_volume': [], 'tick_neco': []}
    # }

    # for key, value in a.items():
    #     if key in b:
    #         filtered_b[key] = {sub_key: b[key][sub_key] for sub_key in b[key] if sub_key in value}



    #TODO na platforme nejak rychlit, init jen jendou
    def column_stack_source(self, state, verbose = 1) -> list[np.array]:
        #pole pro jednotlive vstupy
        input_features = []
        for input_item, settings in self.input.items():
            #print(f"Item: {input_item}")
            #dotahneme delku sekvence 
            sequence_length = settings["sequence_length"]
            features_to_join = []
            for source_key, features in settings.items():
                if source_key not in ["sequence_length", "scaler"]:
                    #dotazeni ze state
                    sources = getattr(state, source_key) 
                    #print(f"{source_key} avilable: {sources.keys()}")
                    poradi_sloupcu = [feature for feature in features]
                    #print("poradi sloupce v source_data", str(poradi_sloupcu))

                    # input_feature = np.column_stack([sources[source_key][feature][-sequence_length:] for feature in  sources[source_key] if feature in features]) 
                    # Optimized list comprehension with padding, if there is not enough data 
                    input_feature = np.column_stack([
                        np.pad(np.array(sources[feature][-sequence_length:]).astype(float), 
                            (max(0, sequence_length - len(sources[feature][-sequence_length:])), 0), 
                            mode='constant', constant_values=np.nan)
                        for feature in features
                    ])
                    features_to_join.append(input_feature)

            #join features with the same time, scale it and add to input list
            joined_features = np.concatenate(features_to_join, axis=1)
            #pokud je pro dany vstup definovan scaler tak scalujeme
            if self.scalersX.get(input_item, None) is not None:
                joined_features = self.scalersX[input_item].transform(joined_features)
            #as padding is done with NaNs, change it to 0 for the model
            joined_features = np.nan_to_num(joined_features)
            #reshape to 3d batch_size(1), sequence_length, feature   ()
            #TODO optimalizovat na speed
            joined_features = joined_features.reshape((1, joined_features.shape[0], joined_features.shape[1]))
            input_features.append(joined_features)
            
        return input_features
    

        #     if "source" in value:
        #         feature_data = []
        #         for source_ in value["source"]:
        #             if source_type in value:
        #                 ## cbar_indicators : ["tick_price", "tick_volume"]
        #                 print(f"{source_type}: {value[source_type]}")
        #                 poradi_sloupcu = [feature for feature in value[source_type] if feature in sources_dict[source_type]]
        #                 print("poradi sloupce v source_data", str(poradi_sloupcu))
        #                 feature_data.append(np.column_stack([sources_dict[source_type][feature] for feature in value[source_type] if feature in sources_dict[source_type]]))

        #         if len(feature_data) >1: 
        #             combined_day_data = np.column_stack(feature_data)
        #         returned_data.append(combined_day_data)
                        
        # return combined_day_data

            # #create SOURCE DATA with features
            # # bars and indicators dictionary and features as input
            # poradi_sloupcu_inds = [feature for feature in self.ind_features if feature in indicators]
            # indicator_data = np.column_stack([indicators[feature] for feature in self.ind_features if feature in indicators])
            
            # if len(bars)>0:
            #     bar_data = np.column_stack([bars[feature] for feature in self.bar_features if feature in bars])
            #     poradi_sloupcu_bars = [feature for feature in self.bar_features if feature in bars]
            #     if verbose == 1:
            #         print("poradi sloupce v source_data", str(poradi_sloupcu_bars + poradi_sloupcu_inds))
            #     combined_day_data = np.column_stack([bar_data,indicator_data])
            # else:
            #     combined_day_data = indicator_data
            #     if verbose == 1:
            #         print("poradi sloupce v source_data", str(poradi_sloupcu_inds))
            # return combined_day_data

    #create TARGET(Y) data 
    def column_stack_target(self, bars, indicators) -> np.array:
        target_base = []
        target_reference = []
        try:
            try:
                target_base = bars[self.target]
            except KeyError:
                target_base = indicators[self.target]
            try:
                target_reference = bars[self.target_reference]
            except KeyError:
                target_reference = indicators[self.target_reference]
        except KeyError:
            pass
        target_day_data = np.column_stack([target_base, target_reference])
        return target_day_data

    def load_runners_as_list(self, runner_id_list = None, batch_id = None):
        """Loads all runners data (bars, indicators) for given runners into list of dicts.
        
        List of runners/train_batch_id may be provided, or self.train_runner_ids/train_batch_id is taken instead.

        Returns:
            Each runner as list item.
        """
        if runner_id_list is not None:
            runner_ids = runner_id_list
            print("loading runners for ",str(runner_id_list))
        elif batch_id is not None:
            print("Loading runners for train_batch_id:", batch_id)
            res, runner_ids = exts.get_archived_runners_list_by_batch_id(batch_id, self.cfg["download_server"])
            if res < 0:
                print("error", runner_ids)
                return None, None
        elif self.train_batch_id is not None:
            print("Loading runners for TRAINING BATCH self.train_batch_id:", self.train_batch_id)
            res, runner_ids = exts.get_archived_runners_list_by_batch_id(self.train_batch_id, self.cfg["download_server"])
            if res < 0:
                print("error", runner_ids)
                return None, None
        #pripadne bereme z listu runneru
        else:
            runner_ids = self.train_runner_ids
            print("loading runners for TRAINING runners ",str(self.train_runner_ids))

        result_list = []

        for runner_id in tqdm(runner_ids):
            daily_dict = defaultdict(list)
            #returns dictionary with keys of distinct_sources
            sources = mu.load_runner(runner_id, self.distinct_sources, self.cfg["download_server"], False)
            print(f"runner:{runner_id}")

            if len(self.distinct_sources) != len(sources):
                raise Exception(f"V runner {runner_id} neni pozadovany pocet zdroju (bars, ind, cbars..) {self.distinct_sources}")

            # for key in self.distinct_sources:
            #     daily_dict[key].append(sources[key])
            
            result_list.append(sources)

        return result_list

    def get_highest_resolution(self):
        max_value = -1
        top_key_with_max_value = None

        for top_key, inner_dict in self.input.items():
            for inner_key in inner_dict.keys():
                if inner_key in SOURCES_GRANULARITY and SOURCES_GRANULARITY[inner_key] > max_value:
                    max_value = SOURCES_GRANULARITY[inner_key]
                    top_key_with_max_value = top_key

        return top_key_with_max_value        

    #NEJSPIS pridat DAY_INDEXING sem
    def prep_data(self, concat_data, day_indexes):

        """
        Creates dataset for each input
            source_dict = {'highres':
                            {'day_index':}
                            {'sampling_time': [1,2,3,4,5,6,7,8,9,10],
                            'sequence_length': 3,
                            'tick_price': [33.67, 33.57, 33.77, 33.74, 33.79, 33.74, 33.74, 33.75, 33.76],
                            'time': [1,2,3,4,5,6,7,8,9,10],
                            'tick_volume': [1,2,3,4,5,6,7,8,9,10]},
                        'lowres':
                            {'sampling_time': [3,5,7,9],
                            'sequence_length': 2,
                            'close': [33.75, 33.815, 33.8, 33.8],
                            'time': [3,5,7,9],
                            'volume': [6499092, 46790, 25000, 14643],
                            'atr10': [0.24, 0.17, 0.1367, 0.1125],
                            'sl_long': [33.27, 33.474999999999994, 33.526599999999995, 33.574999999999996]},
        }

                input = {
                    'highres':
                        {'cbar_indicators': ['tick_price', 'tick_trades', 'tick_volume'],
                        {'sequence_length': 75}
                        {'scaler': "StandardScaler"}
                        
                    'lowres':
                        {'bars': ['close', 'high', 'low', 'open', 'volume'],
                        'indicators': ['atr10', 'sl_long'],
                        'sequence_length': 20},
                        {'scaler': "StandardScaler"}
                    }
        """

        ##bereme vzdy time (ten pak mazeme pokud neni pozadovan) a pak dle nastaveni
        def create_dataset(conf_key, day_indexes):
            dataset = dict(sequence_length=self.input[conf_key]['sequence_length'])

            #iterujeme nad oblastmi a z oblastí vytahneme z nich featury, ktere pozadujeme
            for source_key, features in self.input[conf_key].items():
                if source_key not in ['sequence_length','scaler']:
                    # dataset["areas_included"] = dataset.get("areas_included", []).append(source_key)

                    #pro kazdy vstup pridanem sampling_time a day_index
                    #(i kdyz je vstup slozeny ze dvou oblasti (bars, indicators) tak sampling_time a day_index
                    #bereme vzdy z prvniho - JSOU VZDY STEJNE
                    if dataset.get("sampling_time", None) is None:
                        #time je ve zdrojovych datech vzdy (tbd zmena az se static indikatory)
                        dataset["sampling_time"] = concat_data[source_key]['time']
                    #ulozime day indexy dane oblasti (bars, cbars_indicators ...)
                    if dataset.get("day_indexes", None) is None:
                        dataset["day_indexes"] = [row[source_key] for row in day_indexes]

                    if source_key in concat_data:
                        for feature in features:
                            #pridavame vsechny pozadovane featury z oblasti
                            dataset[feature] = concat_data[source_key][feature]

            return dataset

        
        dataset = {}

        for key in self.input:
                dataset[key] = create_dataset(key, day_indexes)

        return dataset

    #vstupuje cely datase a zaroven index umoznujici daily data [(0,118),(119,250)]
    def create_sequences(self,source, day_indexes):
        X_train = {}
        y_train = []

        #muze mi vytvorit source_dict pro cely dataset
        source_dict = self.prep_data(source, day_indexes)
        #testing data override
        """"
        input = {
        'highres':
            {'cbar_indicators': ['tick_price', 'tick_trades', 'tick_volume'], 
            'sequence_length': 75}
            'scaler': 'XXX'}
        'lowres':
            {'bars': ['close', 'high', 'low', 'open', 'volume'],
            'indicators': ['atr10', 'sl_long'], 
            'sequence_length': 20}
            'scaler': 'XXX'}
        }

        #tento format upravit, aby obsahoval pro kazdy input dalsi pomocny atribut "sampling_time" 
        - podle ktereho se budou zarovnavat sekvence a ktery nebude ve vystupu
        - diky tomu remove_time muze jit pryc

            source_dict = {'highres':
                            {'day_index':}
                            {'sampling_time': [1,2,3,4,5,6,7,8,9,10],
                            'sequence_length': 3,
                            'tick_price': [33.67, 33.57, 33.77, 33.74, 33.79, 33.74, 33.74, 33.75, 33.76],
                            'time': [1,2,3,4,5,6,7,8,9,10],
                            'tick_volume': [1,2,3,4,5,6,7,8,9,10]},
                        'lowres':
                            {'sampling_time': [3,5,7,9],
                            'sequence_length': 2,
                            'close': [33.75, 33.815, 33.8, 33.8],
                            'time': [3,5,7,9],
                            'volume': [6499092, 46790, 25000, 14643],
                            'atr10': [0.24, 0.17, 0.1367, 0.1125],
                            'sl_long': [33.27, 33.474999999999994, 33.526599999999995, 33.574999999999996]},
        }

        pole dennich indexů
         indexes = 
                [{'indicators': (0, 4373), 'bars': (0, 4373), 'cbar_indicators': (0, 64559)}
                [{'indicators': (4373, 8432), 'bars': (4373, 8432), 'cbar_indicators': (64559, 101597)}

        """
        
        #iteratujeme na kazdy den
        #for day_data in tqdm(source):
        for day, indexes in enumerate(day_indexes):

            daily_sequences = ModelML.create_daily_sequences(source_dict, self.highest_res_key, indexes)
            
            for key, sequences in daily_sequences.items():
                if key in X_train:
                    X_train[key] = np.concatenate([X_train[key], sequences], axis=0)
                else:
                    X_train[key] = sequences            

            # Target sequence generation
            #
            #1.If the target is from the highest resolution input: Use the target data as is.
            #2.If the target is from a lower resolution input: Resample the target data to the 
            #   highest resolution by repeating the last known value until a new value is known based on time.
            target_source, target_feature = list(self.target.items())[0]
            if target_source in day_data and target_feature in day_data[target_source]:


                #target_data = source_[target_source][target_feature]
                if target_source in list(source_dict[self.highest_res_key].keys()):
                    # Use target data as is for the highest resolution
                    target_data = source_dict[self.highest_res_key][target_source][target_feature]
                    y_train += target_data
                else:
                    # Resample target data to highest resolution
                    #TODO toto predelat na vecorizaci se zachovanim logiky
                    resampled_target_data = self.resample_to_higher_resolution(day_data, target_source, source_dict, target_feature)
                    y_train += resampled_target_data
            else:
                raise Exception("Target not present")

        y_train = np.array(y_train)
        y_train = y_train.reshape(-1,1) #reshape to (300,1) from (300,)

        return X_train, y_train



    #mozna by stalo za to vytvorit si oba rpistupy (day by day fit and transform)
    def create_sequences_old_working(self,source):
        X_train = {}
        y_train = []
        #iteratujeme na kazdy den
        for day_data in tqdm(source):
            source_dict = self.prep_data(day_data)
            #testing data override
            """"
            input = {
            'highres':
                {'cbar_indicators': ['tick_price', 'tick_trades', 'tick_volume'], 'sequence_length': 75}
            'lowres':
                {'bars': ['close', 'high', 'low', 'open', 'volume'],
                'indicators': ['atr10', 'sl_long'], 'sequence_length': 20}
            }

            source_dict = {'highres':
            {'remove_time': True, 'sequence_length': 3,
             'tick_price': [33.67, 33.57, 33.77, 33.74, 33.79, 33.74, 33.74, 33.75, 33.76],
             'time': [1,2,3,4,5,6,7,8,9,10], 'tick_volume': [1,2,3,4,5,6,7,8,9,10]},
            'lowres':
            {'remove_time': True, 'sequence_length': 2,
             'close': [33.75, 33.815, 33.8, 33.8],
             'time': [3,5,7,9], 'volume': [6499092, 46790, 25000, 14643], 'atr10': [0.24, 0.17, 0.1367, 0.1125], 'sl_long': [33.27, 33.474999999999994, 33.526599999999995, 33.574999999999996]},
            }
            """

            daily_sequences = ModelML.create_daily_sequences(source_dict, self.highest_res_key)
            
            for key, sequences in daily_sequences.items():
                if key in X_train:
                    X_train[key] = np.concatenate([X_train[key], sequences], axis=0)
                else:
                    X_train[key] = sequences            

            # Target sequence generation
            #
            #1.If the target is from the highest resolution input: Use the target data as is.
            #2.If the target is from a lower resolution input: Resample the target data to the 
            #   highest resolution by repeating the last known value until a new value is known based on time.
            target_source, target_feature = list(self.target.items())[0]
            if target_source in day_data and target_feature in day_data[target_source]:


                #target_data = source_[target_source][target_feature]
                if target_source in list(source_dict[self.highest_res_key].keys()):
                    # Use target data as is for the highest resolution
                    target_data = source_dict[self.highest_res_key][target_source][target_feature]
                    y_train += target_data
                else:
                    # Resample target data to highest resolution
                    #TODO toto predelat na vecorizaci se zachovanim logiky
                    resampled_target_data = self.resample_to_higher_resolution(day_data, target_source, source_dict, target_feature)
                    y_train += resampled_target_data
            else:
                raise Exception("Target not present")

        y_train = np.array(y_train)
        y_train = y_train.reshape(-1,1) #reshape to (300,1) from (300,)

        return X_train, y_train

    #TODO sekvencning nyni bere posledni hodnotu (jao u indikatoru)
    #do budoucna upravit sekvencing aby 1:1 odpovidal realu, tzn.
    #v nastaveni si budu moct urcit jako handlovat mezi hodnoty
    #zda interpolaci nebo last value
        
    #NYNI BY DEFAULT LAST VALUE
    #indexes = denni index, 
    
    #muze byt volano z train nebo evaluate, podle toho but i fitujeme nebo jen transformujeme
    #muze byt take volano v ramci batchove pripravy, kdy pouzivame fit partial
    def scale_and_sequence(self, concat_data, day_indexes, scaling: ScalingMode = ScalingMode.FIT_AND_TRANSFORM):
        print(bold("Sequencing started"))
        source_dict = self.prep_data(concat_data, day_indexes)

        """"
        input = {
        'highres':
            {'cbar_indicators': ['tick_price', 'tick_trades', 'tick_volume'], 'sequence_length': 75}
        'lowres':
            {'bars': ['close', 'high', 'low', 'open', 'volume'],
            'indicators': ['atr10', 'sl_long'], 'sequence_length': 20}
        }

        #tento format upravit, aby obsahoval pro kazdy input dalsi pomocny atribut "sampling_time" 
        - podle ktereho se budou zarovnavat sekvence a ktery nebude ve vystupu
        - diky tomu remove_time muze jit pryc

            source_dict = {'highres':
                            {'day_index':}
                            {'sampling_time': [1,2,3,4,5,6,7,8,9,10],
                            'sequence_length': 3,
                            'tick_price': [33.67, 33.57, 33.77, 33.74, 33.79, 33.74, 33.74, 33.75, 33.76],
                            'time': [1,2,3,4,5,6,7,8,9,10],
                            'tick_volume': [1,2,3,4,5,6,7,8,9,10]},
                        'lowres':
                            {'sampling_time': [3,5,7,9],
                            'sequence_length': 2,
                            'close': [33.75, 33.815, 33.8, 33.8],
                            'time': [3,5,7,9],
                            'volume': [6499092, 46790, 25000, 14643],
                            'atr10': [0.24, 0.17, 0.1367, 0.1125],
                            'sl_long': [33.27, 33.474999999999994, 33.526599999999995, 33.574999999999996]},
        }

        pole dennich indexů
         indexes = 
                [{'indicators': (0, 4373), 'bars': (0, 4373), 'cbar_indicators': (0, 64559)}
                [{'indicators': (4373, 8432), 'bars': (4373, 8432), 'cbar_indicators': (64559, 101597)}

        """

        X_train = {}
        y_train = {}
        excluded_keys = ['day_indexes','sampling_time', 'sequence_length']

        #TODO !!!
        #MUSIM SI TO NAKRESLIT NA PORADNY PAPIR A VICE MONITORU

        highest_res_data = source_dict[self.highest_res_key]
        #TODO toto by šlo paralelizovat, je zpracováváno nezávisle
        #zde iterujeme nad inputy (highres, lowres...)
        for key, data in source_dict.items():
            sequence_length = data['sequence_length']
            print(red(f"INPUT: {key}"))
            #vytahneme vsechny featury a jejich data
            features = [np.array(data[feature]) for feature in data if feature not in excluded_keys]
            print("poradi sloupcu ", str([feature for feature in data if feature not in excluded_keys]))
            #TODO not used - jelikoz jsou vsechny featiure ve stejnem sampling rate, pouziju prvni pro nasledne dohledani day indexu
            #first_feature = next((feature for feature in data if feature not in excluded_keys), None)

            #provedeme scaling celeho vstupu a pak tento scalovany vstup a pak tento scalovany vstup opracujeme skrz dny

            
            #print("tady jsou vsechny features pro dany vstup")
            #print("sem by sel iterativni scaler za tento den")
            #print("features pred scalingem", features)

            # Transpose the data so that we have samples as rows and features as columns
            features = np.array(features)
            features = features.T

            #scalerX je nakonfiguruvan per vstup
            if self.scalersX.get(key,None) is not None:
                print(bold(f"Scaling using: {scaling}"))
                match scaling:
                    case ScalingMode.FIT_AND_TRANSFORM:
                        features = self.scalersX[key].fit_transform(features)
                    case ScalingMode.PARTIAL_FIT_AND_TRANSFORM:
                        self.scalersX[key].partial_fit(features)
                        features = self.scalersX[key].transform(features)
                    case ScalingMode.ONLY_TRANSFORM:
                        features = self.scalersX[key].transform(features)
                    case _:
                        raise Exception("unknow scaling method for scalerX")

                print(bold(f"scalered features:{self.scalersX[key].n_features_in_} samples:{self.scalersX[key].n_samples_seen_}"))
            else:
                print(bold(f"NO SCALER defined for {key}"))

            # Transpose Back
            # Converting to a list of three items, each being a numpy array
            features = [features[:, i] for i in range(features.shape[1])]
            #features = [row for row in features]

            #print("features po scalingu", features)
            #scalovany vstup opracujeme na dny
            #pripadne muzeme zapracovat optional iterative scaler

            #tady az teprve zacneme DAY ITEARATION nad indexy tohoto vstupu
            daily_sequences = []
            for day, (start_idx, end_idx) in enumerate(data["day_indexes"]):

                #SEQUENCES per DAY
                #vezmu si pouze denni slice z features start_idx, end_idx
                #CHATGPT - jak to udelat?
                # Splicing each array from start_index to end_index
                spliced_features = [array[start_idx:end_idx] for array in features]

                #uložím si sampling_time nejvyššího rozlišení toho sameho dne a jejich delku
                (start_index_of_highest, end_index_of_higest) = source_dict[self.highest_res_key]["day_indexes"][day]
                highest_res_times = np.array(highest_res_data['sampling_time'][start_index_of_highest:end_index_of_higest])#sem dat dailyslice
                samples = len(highest_res_times)

                # Aligning sequences to the highest resolution
                aligned_indices = np.searchsorted(data['sampling_time'][start_idx:end_idx], highest_res_times, side='right') - 1 #slices
                
                # Creating sequences with padding
                sequences = []
                for i in tqdm(range(samples)):
                    if aligned_indices[i] == -1:
                        sequence = np.zeros((sequence_length, len(spliced_features)))
                    else:
                        start_idx = max(0, aligned_indices[i] - sequence_length + 1)
                        sequence = [feature[start_idx:aligned_indices[i]+1] for feature in spliced_features]
                        sequence = [np.pad(seq, (max(0, sequence_length - len(seq)), 0), mode='constant', constant_values=0) for seq in sequence]
                        sequence = np.stack(sequence, axis=-1)

                    sequences.append(sequence)

                daily_sequences.append(np.array(sequences))

            ##vsechyn denni sekvence daneho klice spojime to je hotove pro dany vstup
            daily_sequences_array = np.concatenate(daily_sequences, axis=0) 
            X_train[key] = daily_sequences_array

        print(red("Sequencing target"))
        ##TADY jeste target - muzeme zpracovat najednou, jen pripadne resamplovat
        # Target sequence generation
        #
        #1.If the target is from the highest resolution input: Use the target data as is.
        #2.If the target is from a lower resolution input: Resample the target data to the 
        #   highest resolution by repeating the last known value until a new value is known based on time.
        target_area, target_feature = list(self.target.items())[0]
        #target_feature (tick_price) exists in highest resolution

        #target_source (cbar_indicators) existuje 
        #TODO sakra to taky nebude fungovat kdyz budu mit ve vstupu jen areu bars a target bude z indikatoru
        #i kdyz maji spolecnou osu, tak ve areas_included nebude sakr

        #pokud je delka poli target casu stejne jako sample_time nejvyssiho rozliseni, target patri sem
        #TODO vymyslet mozna lepe?
        if len(concat_data[target_area]["time"]) == len(source_dict[self.highest_res_key]["sampling_time"]):
            # Use target data as is for the highest resolution
            target_data = concat_data[target_area][target_feature]
            y_train = target_data
        else:
            # Resample target data to highest resolution
            #TODO OTESTOVAT NA MALEM SAMPLU ZDA FUNGUJE i skrz
            #TODO toto predelat na vecorizaci se zachovanim logiky

            #posilame i concat data, protoze tam jedine ted mame target
            resampled_target_data = self.resample_to_higher_resolution(target_area, concat_data, source_dict, target_feature)
            y_train = resampled_target_data

        y_train = np.array(y_train)
        y_train = y_train.reshape(-1,1) #reshape to (300,1) from (300,)
        
        print(bold(f"mode: {scaling}"))
        if self.scalerY is not None:
            match scaling:
                case ScalingMode.FIT_AND_TRANSFORM:
                    y_train = self.scalerY.fit_transform(y_train)
                case ScalingMode.PARTIAL_FIT_AND_TRANSFORM:
                    self.scalerY.partial_fit(y_train)
                    y_train = self.scalerY.transform(y_train)
                case ScalingMode.ONLY_TRANSFORM:
                    y_train = self.scalerY.transform(y_train)
                case _:
                    raise Exception("unknow scaling method")

            scaler = self.scalerY
            print(bold(f"scaled features: {scaler.n_features_in_} samples: {scaler.n_samples_seen_}"))
        return X_train, y_train

    @staticmethod
    def create_daily_sequences_old(source_dict, highest_res_key):
        highest_res_data = source_dict[highest_res_key]
        highest_res_times = np.array(highest_res_data['time'])#sem dat dailyslice
        samples = len(highest_res_times)
        
        output_sequences = {}

        for key, data in source_dict.items():
            sequence_length = data['sequence_length']
            features = [np.array(data[feature]) for feature in data if feature not in ['time','remove_time', 'sequence_length']]#dailyslice
            if not data['remove_time']:
                features.insert(0, np.array(data['time']))#slice
            
            # print("tady jsou vsechny features pro dany vstup")
            # print("sem by sel iterativni scaler za tento den")
            # print(features)

            # # Transpose the data so that we have samples as rows and features as columns
            # features = np.array(features)
            # features = features.T

            # scaler = StandardScaler()
            # # Fitting the scaler to the data and transforming the data
            # features = scaler.fit_transform(features)
            # print("scaler vidi")
            # print("pocet featur:", scaler.n_features_in_)
            # #print(scaler.feature_names_in_)
            # print("pocet samplu:", scaler.n_samples_seen_)

            # # Transpose Back
            # features = features.T
            # features = features.tolist()

            # Aligning sequences to the highest resolution
            aligned_indices = np.searchsorted(data['time'], highest_res_times, side='right') - 1 #slices
            
            # Creating sequences with padding
            sequences = []
            for i in tqdm(range(samples)):
                if aligned_indices[i] == -1:
                    sequence = np.zeros((sequence_length, len(features)))
                else:
                    start_idx = max(0, aligned_indices[i] - sequence_length + 1)
                    sequence = [feature[start_idx:aligned_indices[i]+1] for feature in features]
                    sequence = [np.pad(seq, (max(0, sequence_length - len(seq)), 0), mode='constant') for seq in sequence]
                    sequence = np.stack(sequence, axis=-1)

                sequences.append(sequence)

            output_sequences[key] = np.array(sequences)
        
        return output_sequences

    def resample_to_higher_resolution(self, target_source, concat_data, source_dict, target_feature):
        # Resample target data to the time scale of the highest resolution
        target_time_data = concat_data[target_source]['time']
        highest_res_time_data = source_dict[self.highest_res_key]['sampling_time']
        resampled_data = []
        target_index = 0

        for time_point in tqdm(highest_res_time_data):
            while target_index + 1 < len(target_time_data) and target_time_data[target_index + 1] <= time_point:
                target_index += 1
            resampled_data.append(concat_data[target_source][target_feature][target_index])

        return resampled_data

    def resample_to_higher_resolution_old(self, day_data, target_source, source_dict, target_feature):
        # Resample target data to the time scale of the highest resolution
        target_time_data = day_data[target_source]['time']
        highest_res_time_data = source_dict[self.highest_res_key]['time']
        resampled_data = []
        target_index = 0

        for time_point in tqdm(highest_res_time_data):
            while target_index + 1 < len(target_time_data) and target_time_data[target_index + 1] <= time_point:
                target_index += 1
            resampled_data.append(day_data[target_source][target_feature][target_index])

        return resampled_data

    def create_sequence_for_time(self, data, time_data, current_time, seq_length):
        # Create a sequence based on time comparison
        indices = [i for i, t in enumerate(time_data) if t <= current_time]
        if len(indices) < seq_length:
            pad_size = seq_length - len(indices)
            padded_data = [0] * pad_size + [data[i] for i in indices]
            return np.array(padded_data).reshape(-1, 1)
        else:
            selected_indices = indices[-seq_length:]
            return np.array([data[i] for i in selected_indices]).reshape(-1, 1)

    #vytvori X a Y data z nastaveni self
    #pro vybrane runnery stahne data, vybere sloupce dle faature a target
    #a vrátí jako sloupce v numpy poli
    #zaroven vraci i rows_in_day pro nasledny sekvencing
    #kdyz neplnime vstup, automaticky se loaduje training data z nastaveni classy
    def load_data(self, runners_ids: list = None, batch_id: list = None, source: Source = Source.RUNNERS):
        """Service to load data for the model. Can be used for training or for vector prediction.

        If input data are not provided it falls back to the class init self.cfguration (train_runners_ids, train_batch_id)

        Args:
            runner_ids: 
            batch_id:
            source: To load sample data.

        Returns:
            source_data,target_data,rows_in_day
        """
        rows_in_day = defaultdict(list)
        indicatorslist = []
        sources_dict = defaultdict(list)
        #bud natahneme samply

        #TODO toto nejspis pryc
        if source == Source.SAMPLES:
            if self.use_bars:
                bars = sample_bars
            else:
                bars = {}
            indicators = sample_indicators
            indicatorslist.append(indicators)
        #nebo dotahneme pozadovane runnery
        else:


            #nalodujeme vsechny runnery jako listy (bud z runnerids nebo dle batchid)
            #mame vsechny atributy
            #vracime celkovy list runnerů, díky rozdělení runnerů už máme crossday sekvence, tzn. budeme sekvencovat po runnerech
            sources = self.load_runners_as_list(runner_id_list=runners_ids, batch_id=batch_id)

            # src = deepcopy(sources)
            # #zmergujeme jejich data dohromady
            # for key in distinct_sources:
            #     sources_dict[key] = mu.merge_dicts(src[key])

        print(red(f"LOADED {len(sources)} days"))
        print("Distinct sources:", self.distinct_sources)
        for idx, value in enumerate(sources):
            print(bold(f"Day: {idx+1}"))
            for key in self.distinct_sources: 
                klice = sources[idx][key].keys() 
                first_key_idx = list(klice)[0]
                first_key_len = len(sources[idx][key][first_key_idx])     
                print(f"{key}: {len(sources[idx][key])} length: {first_key_len}")


        #pritomnost targetu validovat pozdeji
        self.validate_available_features(sources)    
        return sources
        # ##tady to vratime
        # #a stacking udelame az v sequencingu, kvuli casove souslednosti

        # print("Preparing FEATURES")
        # source_data, target_data = self.stack_source_data(sources)
        # return source_data, target_data, rows_in_day

    #list1 is bigger
    @staticmethod
    def list1_subset_list2(list1, list2):
        return set(list2).issubset(set(list1))

    def validate_available_features(self, sources_dict):
        for idx, value in enumerate(sources_dict):
            for key, feature_list in self.features_required.items():
                if ModelML.list1_subset_list2(sources_dict[idx][key].keys(),list(feature_list)) is False:
                    print(f"Missing features in '{key}' required {feature_list} but in day {idx} are these: {sources_dict[idx][key].keys()}")
                    print(f"REQ:{feature_list}")
                    print(f"CUR:{sources_dict[idx][key].keys()}")
                    raise Exception(f"Missing features in {key} required {feature_list} but in day {idx} found only these: {sources_dict[idx][key].keys()}")

    def validate_available_features_old(self, bars, indicators):
        for k in self.bar_features:
            if not k in bars.keys():
                raise Exception(f"Missing bar feature {k}")

        for k in self.ind_features:
            if not k in indicators.keys():
                raise Exception(f"Missing ind feature {k}")    

    def stack_source_data(self, sources_dict):
        print("Stacking dicts to numpy")
        print("Source - X")
        source_data = self.column_stack_source(sources_dict)
        print("shape", np.shape(source_data))
        print("Target - Y", self.target)
        target_data = self.column_stack_target(sources_dict)
        print("shape", np.shape(target_data))

        return source_data, target_data

    #pomocna sluzba, ktera provede vsechny transformace a inverzni scaling  a vyleze z nej predikce
    #vstupem je standardni format ve strategii (state.bars, state.indicators)
    #vystupem je jedna hodnota

    """
      input = {  
      'highres':
        {'cbar_indicators': ['tick_price', 'tick_trades', 'tick_volume'], 'sequence_length': 75},
      'lowres':
        {'bars': ['close', 'high', 'low', 'open', 'volume'], 'indicators': ['atr10', 'sl_long'], 'sequence_length': 20},
        }
    """


    #SKALAR PREDICT
    #TODO toto zefektivnit a zhomogenizovat s TRAINEM - MUSI POUZIVAT STEJNE TOOLY
    def predict(self, sources) -> float:
        transf_input = self.column_stack_source(sources, verbose=0)
        # Make a prediction
        #TODO porovnat performance s predict_on_batch
        #start_time = time.time() 
        prediction = self.model.predict_on_batch(transf_input)
        if self.scalerY is not None:
            prediction = self.scalerY.inverse_transform(prediction)
        #end_time = time.time() 
        #print(f"TIME: {end_time-start_time}s")
        return prediction

    def plot_target(self, y_train,y_train_ref):   #zobrazime si transformovany target a jeho referncni sloupec
        #ZHOMOGENIZOVAT OSY
        plt.plot(y_train, label='Transf target')
        plt.plot(y_train_ref, label='Ref target')
        plt.plot()
        plt.legend()
        plt.savefig("res_target.png")
        #plt.show()