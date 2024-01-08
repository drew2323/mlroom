import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlroom.utils.mlutils as mu
from mlroom.utils.mlutils import red, bold
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
#import matplotlib
#atplotlib.use('TkAgg')  # Use an interactive backend like 'TkAgg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt
from mlroom.ml import ModelML
from mlroom.utils.enums import PredOutput, Source, TargetTRFM, ScalingMode
import mlroom.arch.architectures as ma
from mlroom.config import DEF_CACHE_FILE,load_config
# from collections import defaultdict
# from operator import itemgetter
from joblib import load, dump
import argparse
import toml
import time

# CONFIG, CONFIG_STRING = {}, None
# region Notes

#ZAKLAD PRO TRAINING SCRIPT na vytvareni model u
# TODO
# podpora pro BINARY TARGET
# podpora hyperpamaetru (activ.funkce sigmoid atp.)
# vyuzit distribuovane prostredi - nebo aspon vlastni VM
# dopracovat denni identifikatory typu lastday close, todays open atp.
# random SEARCH a grid search
# udelat nejaka model metadata (napr, trenovano na (runners+obdobi), nastaveni treningovych dat, počet epoch, hyperparametry, config atribu atp.) - mozna persistovat v db
# udelat nejake verzovani
# predelat do GUI a modulu
# vyuzit VectorBT na dohledani optimalizovanych parametru napr. pro buy,sell atp. Vyuzit podobne API na pripravu dat jako model.
# EVAL MODEL - umoznit vektorové přidání indikátoru do runneru (např. predikce v modulu, vectorBT, optimalizace atp) - vytvorit si na to API, podobne co mam, nacte runner, transformuje, sekvencuje, provede a pak zpetne transformuje a prida jako dalsi indikator. Lze pak použít i v gui.
# nove tlacitko "Display model prediction" na urovni archrunnera, které
#   - má volbu model + jestli zobrazit jen predictionu jako novy indikator nebo i mse from ytarget  (nutny i target)
# po spusteni pak:
#   - zkonztoluje jestli runner ma indikatory,ktere odpovidaji features modelu (bar_ftrs, ind_ftrs, optional i target)
#   - vektorově doplní predictionu (transformuje data, udela predictionu a Y transformuje zpet)
#    - vysledek (jako nove indikatory) implantuje do runnerdetailu a zobrazi
# podivat se na dalsi parametry kerasu, napr. false positive atp.
# podivat se jeste na rozdil mezi vectorovou predikci a skalarni - proc je nekdy rozdil, odtrasovat - pripadne pogooglit
#      odtrasovat, nekde je sum (zkusit si oboji v jednom skriptu a porovnat)

#TODO NAPADY Na modely
#1.binary identifikace trendu napr. pokud nasledujici 3 bary rostou (0-1) nebo nasledujici bary roste momentum
#2.soustredit se na modely s vystupem 0-1 nebo -1 až 1
#3.Vyzkouset jeden model, ktery by identifikoval trendy v obou smerech - -1 pro klesani a 1 pro stoupání.
#4.vyzkouset zda model vytvoreny z casti dne nebude funkcni na druhe casti (on the fly daily models)
#5.zkusit modely s a bez time (prizpusobit tomu kod v ModelML - zejmena jak na crossday sekvence) - mozna ze zecatku dat aspon pryc z indikatoru? 
# Dat vsechny zbytecne features pryc, nechat tam jen ty podstatne - attention, tak cílím.
#6. zkusit vyuzit tickprice v nejaekm modelu, pripadne pak dalsi CBAR indikatory . vymslet tickbased features
#7. zkusit jako features nevyuzit standardni ceny, ale pouze indikatory reprezentujici chovani (fastslope,samebarslope,volume,tradencnt)
#8. relativni OHLC -  model pouzivajici (jen) bary, ale misto hodnot ohlc udelat features reprezentujici vztahy(pomery) mezi temito velicinami. tzn. relativni ohlc
#9. jiny pristup by byl ucit model na konkretnich  chunkach, ktere chci aby mi identifikoval. Např. určité úseky. Vymyslet. Buď nyni jako test intervaly, ale v budoucnu to treba jen nejak oznacit a poslat k nauceni. Pripadne pak udelat nejaky vycuc.
#10. mozna správným výběrem targetu, můžu taky naučit jen určité věci. Specializace. Stačí když se jednou dvakrát denně aktivuje.
# 11. udelat si go IN model, ktery pomuze strategii generovat vstup - staci jen aby mel trochu lepsi edge nez conditiony, o zbytek se postara logika strategie
# 12. model pro neagregované nebo jen filtroné či velmi lehce agregované trady? - tickprice
# 13. jako featury pouzit Fourierovo transformaci, na sekundovem baru nebo tickprice

#DULEZITE
# soustredit se v modelech na predikci nasledujici hodnoty, ideálně nějaký vektor ukazující směr (např. 0 - 1, kde nula nebude růst, 1 - bude růst strmě)
# pro predikcí nějakého většího trendu, zkusti více modelů na různých rozlišení, každý ukazuje
# hodnotu na svém rozlišení a jeho kombinace mi může určit vstup. Zkusit zda by nešel i jeden model.
# Každopádně se soustředit 
# 1) na další hodnotu (tzn. vstupy musí být bezprostředně ovlivňující tuto (samebasrlope, atp.))
# 2) její výše ukazuje směr na tomto rozlišení
# 3) ideálně se učit z každého baru, tzn. cílová hodnota musí být známá u každého baru 
#      (binary ne, potřebuju linární vektor) -  i když 1 a 0 target v závislosti na stoupání a klesání by mohla být ok, 
#       ale asi příliš restriktivní, spíš bych tam mohl dát jak moc. Tzn. +0.32, -0.04. Učilo by se to míru stoupání.
#       Tu míru tam potřebuju zachovanou.
# pak si muzu rict, když je urcite pravdepodobnost, ze to bude stoupat (tzn. dalsi hodnota) na urovni 1,2,3 - tak jduvstup
# zkusit na nejnižší úrovni i předvídat CBARy, směr dalšího ticku. Vyzkoušet.

##TODO - doma
#bar_features a ind_features do dokumentace SL classic, stejne tak conditional indikator a mathop indikator
#TODO - co je třeba vyvinout
# GENERATOR test intervalu (vstup name, note, od,do,step)
# napsat API, doma pak simple GUI
# vyuziti ATR (jako hranice historickeho rozsahu) - atr-up, atr-down
#    nakreslit v grafu atru = close+atr, atrd = close-atr
#    pripadne si vypocet atr nejak customizovat, prip. ruzne multiplikatory pro high low, pripadne si to vypocist podle sebe
#    vyuziti:
#        pro prekroceni nejake lajny, napr. ema nebo yesterdayclose
#               - k identifikaci ze se pohybuje v jejim rozsahu
#              - proste je to buffer, ktery musi byt prekonan, aby byla urcita akce
#        pro learning pro vypocet conditional parametru (1,0,-1) prekroceni napr. dailyopen, yesterdayclose, gapclose
#             kde 1 prekroceno, 0 v rozsahu (atr), -1 prekroceno dolu - to pomuze uceni
#       vlastni supertrend strateige
#       zaroven moznost vyuzit klouzave či parametrizovane atr, které se na základě
#       určitých parametrů bude samo upravovat a cíleně vybočovat z KONTRA frekvencí, např. randomizovaný multiplier nebo nejak jinak ovlivneny minulým
# v indikatorech vsude kde je odkaz ma source jako hodnotu tak defaultne mit moznost uvest lookback, napr. bude treba porovnavat nejak cenu vs predposledni hodnotu ATRka (nechat az vyvstane pozadavek)
# zacit doma na ATRku si postavit supertrend, viz pinescript na ploše


#TODO - obecne vylepsovaky
# 1. v GUI graf container do n-TABů, mozna i draggable order, zaviratelne na Xko (innerContainer)
# 2. mit mozna specialni mod na pripravu dat (agreg+indikator, tzn. vse jen bez vstupů) - můžu pak zapracovat víc vectorové doplňování dat
#      TOTO:: mozna by postacil vypnout backtester (tzn. no trades) - a projet jen indikatory. mozna by slo i vectorove optimalizovat.
#      indikatory by se mohli predsunout pred next a next by se vubec nemusel volat (jen nekompatibilita s predch.strategiemi)
# 3. kombinace fastslope na fibonacci delkach (1,2,3,5..) jako dobry vstup pro ML
# 4. podivat se na attention based LSTM zda je v kerasu implementace
# do grafu přidat togglovatelné hranice barů určitých rozlišení - což mi jen udělá čáry Xs od sebe (dobré pro navrhování)
# 5. vymyslet optimalizovane vyuziti modelu na produkci (nejak mit zkompilovane, aby to bylo raketově pro skalár) - nyní to backtest zpomalí 4x
# 6. CONVNETS for time series forecasting - small 1D convnets can offer a fast alternative to RNNs for simple tasks such as text classification and timeseries forecasting.
#     zkusit small conv1D pro identifikaci víření před trendem, např. jen 6 barů - identifikovat dobře target, musí jít o tutovku na targetu
#     pro covnet zkusit cbar price, volume a time. Třeba to zachytí víření (ripples)
# Další oblasti k predikci jsou ripples, vlnky - předzvěst nějakého mocnějšího pohybu. A je pravda, že předtím se mohou objevit nějaké indicie. Ty zkus zachytit.
# Do runner_headers pridat bt_from, bt_to - pro razeni order_by, aby se runnery vzdy vraceli vzestupne dle data (pro machine l)

#TODO
# vyvoj modelů workflow s LSTMtrain.py
# 1) POC - pouze zde ve skriptu, nad 1-2 runnery, okamžité zobrazení v plotu,
#           optimalizace zakl. features a hyperparams. Zobrazit i u binary nejak cenu.
# 2) REALITY CHECK - trening modelu na batchi test intervalu, overeni ve strategii v BT, zobrazeni predikce v RT chartu
# 3) FINAL TRAINING
# testovani predikce


#TODO tady
# train model
#     - train data-  batch nebo runners
#     - test data  - batch or runners (s cim porovnavat/validovat)
#     - vyber architektury
#     - soucast skriptu muze byt i porovnavacka pripadne nejaky search optimalnich parametru

#lstmtrain - podporit jednotlive kroky vyse
#modelML - udelat lepsi PODMINKY
#frontend? ma cenu? asi ano - GUI na model - new - train/retrain-change
#  (vymyslet jak v gui chytře vybírat arch modelu a hyperparams, loss, optim - treba nejaka templata?)
#   mozna ciselnik architektur s editačním polem pro kód -jen pár řádků(.add, .compile) přidat v editoru
#    vymyslet jak to udělat pythonově
#testlist generator api

# endregion
def main(mode, to_file = None, from_file = None, config_file = None):
    #load TOML (provided or default)
    global CONFIG, CONFIG_STRING
    CONFIG, CONFIG_STRING = load_config(config_file)
    """
    run.py train
    run.py train --to_file input_data.joblib
    run.py train --from_file input_data.joblib
    run.py prepare --to_file input_data.joblib
    """
    hub(mode, to_file, from_file)

#TODO vyzkouset jak se vysledek bude lisit pri pouzivani batchu a partial fitu
#POZOR pri batchi je dulezite, aby kazda batch mela reprezentativní vzorek, zejména target, aby obsahoval vsechny hodnoty
def train_batch(model_instance: ModelML, source_data: list, mode, to_file, batch_number = 1, total_batches = 1):
    concatenated, day_indexes = mu.concatenate_loaded_data(source_data)

    #scaling mode (if 1 batch use fit_and_transofrm, if more batches - use partial)
    if batch_number == 1 and total_batches == 1:
        scaling = ScalingMode.FIT_AND_TRANSFORM
    else:
        scaling = ScalingMode.PARTIAL_FIT_AND_TRANSFORM

    X_train, y_train = model_instance.scale_and_sequence(concatenated, day_indexes, scaling)

    #zatim pouzity stejny SCALER, v budoucnu vyzkouset vyuziti separatnich scalu pro kazde
    #rozliseni jak je naznaceno zde: https://chat.openai.com/c/2206ed8b-1c97-4970-946a-666dcefb77b4
    #print(f"{X_train}")
    #print(f"{y_train}")

    X_train = list(X_train.values())

    print(red("Post sequencing summary"))
    print("X_train")
    for idx, x in enumerate(X_train):
        print(red(f"input: {idx} {np.shape(x)}"))
    
    print(red(f"y_train: {np.shape(y_train)}"))
  
    source_data = None

    test_size = float(CONFIG["validation"].get("test_size",0))

    #nechame si takhle rozdelit i referencni sloupec
    y_train_ref = np.array([], dtype=np.float64)

    #SPLITTING TRAINING DATA - DECOMM - validacni data budou separatne
    if test_size > 0:
        # Split the data into training and test sets - kazdy vstupni pole rozdeli na dve
        *X_train, y_train, y_test = train_test_split(*X_train, y_train, test_size=test_size, shuffle=False) #random_state=42)

        #X_train je multiinput - nyni obsahuje i train(v sudych) i test(lichych) - rozdelim
        X_test = [element for index, element in enumerate(X_train) if index % 2 != 0]
        X_train = [element for index, element in enumerate(X_train) if index % 2 == 0]

        print("Splittig the data")
        print("X_train")
        for idx, x in enumerate(X_train):
            print("input:",idx)
            print(f"source:X_train[{idx}]", np.shape(x))

        for idx, x in enumerate(X_test):
                print("input:",idx)
                print(f"source:X_test[{idx}]", np.shape(x))

        print("y_train", np.shape(y_train))
        print("y_test", np.shape(y_test))

    validation_tuple = None
    if CONFIG.get("validation",{}).get("validate_during_fit", False) is True:
        #naloadujeme bud runner nebo batch a posleme do treninku
        validation_tuple = model_instance.load_validation_data()

    #SAVE CACHED DATA

    if to_file is not None:
        save_cache(file_name=to_file, X_train=X_train,y_train=y_train,batch_number=batch_number,total_batches=total_batches,validation_tuple=validation_tuple, scalerY=model_instance.scalerY, scalersX=model_instance.scalersX)

    if mode == "prepare":
        print("Data prepared.")
        return

    #TRAIN and SAVE/UPLOAD - train the model and save or upload it according to cfg
    model_instance.train_and_store(X_train, y_train, batch_number, total_batches, validation_tuple)

    print(red(f"Batch {batch_number}/{total_batches} TRAINGING FINISHED"))
    #VALIDATION PART

def save_cache(file_name, **dict_to_save):
    # Save the data to a file
    file_name = DEF_CACHE_FILE if file_name is None else file_name
    dump(dict_to_save, file_name, compress=9)
    print(bold(f"Data saved to {file_name}"))

def load_cache(file_name):
    return load(file_name)

def hub(mode, to_file, from_file):
    validation_runners = CONFIG.get("validation",{}).get("runners", None)
    validation_batch = CONFIG.get("validation",{}).get("batch", None)
    np.set_printoptions(threshold=10,edgeitems=5)

    model_instance = ModelML(cfg=CONFIG, cfg_toml=CONFIG_STRING)
    
    #ulozeny argumenty z cli
    model_instance.metadata["history"]["args"] = args

    #to_file is present = prepare/train with cache storing
    if from_file is None:
        #Loads training data (by distinct_sources)
        source_data = model_instance.load_data() #per day as list
        src_length = len(source_data) 
        items_in_batch = model_instance.train_runners_per_batch
        print("Runners per batch:",items_in_batch)
        items_in_batch = src_length if items_in_batch is None or items_in_batch>src_length else items_in_batch
        total_batches = -(-src_length // items_in_batch)  # Calculate the total number of batches

        # NOTE Batch zatim nepouzivat, vnitrni casti jako cache uz s tim nepocitaji a pro trenink je lepsi vse v jednom
        print("Number of days requested",src_length)
        #print("Iterate to",items_in_batch)
        for i in range(0, len(source_data), items_in_batch):
            batch_number = i // items_in_batch + 1
            print(f"Batch number {batch_number}/{total_batches}")
            print(f"Items {i} to {i+items_in_batch}")
            batch_source_data = source_data[i:i + items_in_batch]
            # Process the batch here
            train_batch(model_instance, batch_source_data, mode, to_file, batch_number, total_batches)
    #from_file is present (its training with loading from cache)
    else:
        cache = load_cache(from_file)
        if cache is None:
            print("ERROR LOADING CACHE")
            return 
        print(red(f"Loaded DATA from cache {from_file}"))
        X_train = cache["X_train"]
        y_train = cache["y_train"]
        batch_number = cache["batch_number"]
        total_batches = cache["total_batches"]
        validation_tuple = cache["validation_tuple"]
        model_instance.scalerY = cache["scalerY"]
        model_instance.scalersX = cache["scalersX"]

        if mode == "train":
            #TRAIN and SAVE/UPLOAD - train the model and save or upload it according to cfg
            model_instance.train_and_store(X_train, y_train, batch_number, total_batches, validation_tuple)
            print(f"Batch {batch_number}/{total_batches} TRAINGING FINISHED")

    if mode == "prepare":
        print("Data preparation finished")
        return

    print(red("STARTING VALIDATION"))
    model_instance: ModelML = mu.load_model(model_instance.name, model_instance.version)
    if (validation_runners is not None and len(validation_runners) > 0) or validation_batch is not None:
        predict_live(model_instance, validation_runners, validation_batch)

        res  = model_instance.load_validation_data()
        if res is not None:
            X_test, y_test = res
            vector_evaluation(model_instance, X_test, y_test)
            measure_infer_speed(model_instance, X_test)


def vector_evaluation(model_instance, X_test, y_test):
    print(bold("STARTING EVALUATION - VECTOR BASED"))
    result = model_instance.model.evaluate(X_test, y_test)
    print(result)

def measure_infer_speed(model_instance, X_test):
    print(bold("ITERATIONs - to measure INFER SPEED"))
    total_time = 0
    for idx, val in enumerate(X_test[0]):
        one_sample = []
        for sample in X_test:
            # Reshape the sample if necessary to match LSTM input requirements
            # E.g., sample might need to be reshaped to (1, time_steps, features)
            one_sample.append(sample[idx].reshape(1, -1, sample[idx].shape[-1]))

        start_time = time.time()  # Start timing
        # Feed the reshaped sample to your LSTM model
        prediction = model_instance.model.predict_on_batch(one_sample)
        #prediction = model_instance.model(one_sample, training=False)
        if model_instance.scalerY is not None:
            prediction = model_instance.scalerY.inverse_transform(prediction)
        #print(prediction)
        end_time = time.time()  # End timing
        #print("val:", prediction)
        #print(f"IT time: {end_time - start_time} seconds")
        iteration_time = end_time - start_time
        total_time += iteration_time

    average_time = total_time / len(X_test[0])
    print(f"Average time per iteration: {average_time} seconds")

def predict_live(model_instance, validation_runners, validation_batch):
    print(red("SCALAR PREDICTION -LIVE SIMULATION"))
    #EVALUATE SIM LIVE - PREDICT SCALAR - based on last X items
    sources = model_instance.load_runners_as_list(runner_id_list=validation_runners, batch_id=validation_batch)
    #zmergujeme vsechny data dohromady 
    model_instance.validate_available_features(sources)

    #VSTUPEM JE dict(indicators=[],bars=[],cbar_indicators[], dailyBars=[]) - nebo jen state?
    # Dynamically create a state class
    start_time = time.time()  # Start timing
    State = type('State', (object,), {**sources[0]})
    # Create an instance of the dynamically created class
    state = State()   
    value = model_instance.predict(state)
    print(red("prediction for LIVE SIM:", value))
    print("Shape of predictions", value.shape)
    if value.shape == (1,1):
        print("Value:",float(value))
    else:
        print("predicted max", np.argmax(value, axis=1))

    end_time = time.time()  # End timing
    print(bold(f"Time taken for this iteration: {end_time - start_time} seconds"))

if __name__ == "__main__":
    """
    Prepare from inputs and train   
    - run.py train
    - run.py train --toml custom.toml 

    Prepare from inputs, store to cache and train
    - run.py train --to_file input_data.joblib

    Load from cache and train
    - run.py train --from_file input_data.joblib

    Prepare from inputs and store to cache
    run.py prepare --to_file input_data.joblib
    """
    parser = argparse.ArgumentParser(description="Process different modes of operation.")

    # Define a positional argument for mode
    parser.add_argument("mode", choices=['train', 'prepare'], help="Mode of operation: train or prepare")

    # Define an optional argument for the cache file
    parser.add_argument("--to_file", help="Cache file to store prepared data.", default=None)

    parser.add_argument("--from_file", help="CAche file to load data from", default=None)

    parser.add_argument("--toml", help="TOML configuration file", default=None)

    args = parser.parse_args()

    main(args.mode, args.to_file, args.from_file, args.toml)