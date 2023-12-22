import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlroom.utils.mlutils as mu
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
#import matplotlib
#atplotlib.use('TkAgg')  # Use an interactive backend like 'TkAgg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt
from mlroom.ml import ModelML
from mlroom.utils.enums import PredOutput, Source, TargetTRFM
import mlroom.arch.architectures as ma
from mlroom.config import CONFIG, CONFIG_STRING
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
import argparse
import toml
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
def main():
    train()

def train():
    #if null,the validation is made on 10% of train data
    #runnery pro testovani
    validation_runners = CONFIG["validation"]["runners"]

    #u binary bude target bud hotovy indikator a nebo jej vytvorit on the fly
    #mozna sem dal pluginovy eval load model architektury 
    #CHTELO BY TO SEM i ulozit nastaveni architektury se kterou se testuje

    model_instance = ModelML(**CONFIG["model"], cfg=CONFIG, cfg_toml=CONFIG_STRING)
    source_data = model_instance.load_data()

    # if len(target_data) == 0:
    #     raise Exception("target is empty - required for TRAINING - check target column name")

    #vytvorime si prepdata ve formatu
    # highres = dict(time, feat1..N, seq_length)
    # lowre = [(dict(time, feat1..N, seq_length)]

    np.set_printoptions(threshold=10,edgeitems=5)
   # print("source_data",source_data)

    #print("source_data", source_data)

    #nejspis az po sekvencingu?
    #source_data = model_instance.scalerX.fit_transform(source_data)

    #vytvořeni sekvenci po vstupních sadách  (např. 10 barů) - výstup 3D např. #X_train (6205, 10, 14)
    #doplneni transformace target data
    #v případě multi-inputu je X_train pole vstupnich sad

    # X_train, y_train, y_train_ref 
    #data = model_instance.preprocess(source_data)


    X_train, y_train = model_instance.create_sequences(source_data)

    #TODO scaling idealne po zakladni transformaci a pred sequencingem




    #TODO zatim pouzity stejny SCALER, v budoucnu vyzkouset vyuziti separatnich scalu pro kazde
    #rozliseni jak je naznaceno zde: https://chat.openai.com/c/2206ed8b-1c97-4970-946a-666dcefb77b4
    print(f"{X_train}")
    print(f"{y_train}")

    X_train = list(X_train.values())

    print("After sequencing")
    print("X_train")
    for idx, x in enumerate(X_train):
        print("input:",idx)
        print("source:X_train", np.shape(x))
    
    print("target:y_train", np.shape(y_train))
  
    #USING SINGLE SCALER
    # Flatten each input
    flattened_inputs = [input.reshape(input.shape[0], -1) for input in X_train]

    # Concatenate flattened inputs
    concatenated_inputs = np.hstack(flattened_inputs)    
    # Fit the scaler on the concatenated data
    model_instance.scalerX.fit(concatenated_inputs)
    
    # Transform the entire concatenated input
    transformed_inputs = model_instance.scalerX.transform(concatenated_inputs)

    #tzn. kdyz budu predikovat musim si si transformovat pole do (,270) - transformovat a pak zase zpatky .. hmm
    #nebude lepsi tri scalery?

    # Split and reshape
    scaled_X_train = []
    start_idx = 0
    for original_shape in [x.shape for x in X_train]:
        end_idx = start_idx + original_shape[1] * original_shape[2]
        # Extract the specific section of transformed_inputs
        transformed_section = transformed_inputs[:, start_idx:end_idx]
        reshaped_input = transformed_section.reshape(original_shape)
        scaled_X_train.append(reshaped_input)
        start_idx = end_idx

    source_data = None
    target_data = None

    #zobrazit target if necessary
    #model_instance.plot_target(y_train,y_train_ref)

    print("After SCALING retransform")
    print("X_train")
    for idx, x in enumerate(scaled_X_train):
        print("input:",idx)
        print("source:X_train", np.shape(x))
    print("target:", y_train)

    X_train = scaled_X_train


    #target scaluji az po transformaci v create sequence -narozdil od X je stejny shape
    y_train = model_instance.scalerY.fit_transform(y_train)

    test_size = None
    if "test_size" in CONFIG["validation"]:
        if CONFIG["validation"]["test_size"] != 0:
            test_size = float(CONFIG["validation"]["test_size"])

    # Split the data into training and test sets - kazdy vstupni pole rozdeli na dve
    #nechame si takhle rozdelit i referencni sloupec
    y_train_ref = np.array([], dtype=np.float64)

    #y_train_ref nyni pryc
    #X_splitted = train_test_split(*X_train, y_train, y_train_ref, test_size=test_size, shuffle=False) #random_state=42)
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
    # print("y_test_ref", np.shape(y_test_ref))
    # print("y_train_ref", np.shape(y_train_ref))

    #print(np.shape(X_train))
    #TRAIN and SAVE/UPLOAD - train the model and save or upload it according to cfg
    model_instance.train_and_store(X_train, y_train)

    print("TRAINGING FINISHED")
    #VALIDATION PART

    print("STARTING VALIDATION")
    #TBD db layer
    model_instance: ModelML = mu.load_model(model_instance.name, model_instance.version)

    # region Live predict
    if len(validation_runners) > 0:
        #EVALUATE SIM LIVE - PREDICT SCALAR - based on last X items
        sources = model_instance.load_runners_as_list(runner_id_list=validation_runners)
        #zmergujeme vsechny data dohromady 
        model_instance.validate_available_features(sources)

        #sources jako pole jednotlivych dnu

        #pro skalar predict potrebujeme jen jako jednu?

        #VSTUPEM JE dict(indicators=[],bars=[],cbar_indicators[], dailyBars=[]) - nebo jen state?
        value = model_instance.predict(sources[0])
        print("prediction for LIVE SIM:", value)
        # endregion

    #EVALUATE TEST DATA - VECTOR BASED
    #pokud mame eval runners pouzijeme ty, jinak bereme cast z testovacich dat
    validation_batch = CONFIG["validation"]["batch"] if "batch" in CONFIG["validation"] else None
    if len(validation_runners) > 0 or validation_batch is not None:
        print(f"Loading validations {validation_runners=} {validation_batch=}")
        source_data, target_data, rows_in_day = model_instance.load_data(runners_ids=validation_runners, batch_id=validation_batch)
        source_data = model_instance.scalerX.fit_transform(source_data)
        X_test, y_test, y_test_ref = model_instance.create_sequences(combined_data=source_data, target_data=target_data,remove_cross_sequences=True, rows_in_day=rows_in_day)
    else:
        print("For validation part of testdata is used", test_size)
    #prepnout ZDE pokud testovat cely bundle - jinak testujeme jen neznama
    #X_test = X_complete
    #y_test = Y_complete

    #toto zmenit na keras 3.0 EVALUATE

    X_test = model_instance.model.predict(X_test)
    X_test = model_instance.scalerY.inverse_transform(X_test)

    #target testovacim dat proc tu je reshape?
    #y_test.reshape(-1, 1)
    y_test =  model_instance.scalerY.inverse_transform(y_test.reshape(-1, 1))
    #celkovy mean? nebo spis vector pro graf?
    mse = mean_squared_error(y_test, X_test)
    print('Test MSE:', mse)

    # Plot the predicted vs. actual
    plt.plot(y_test, label='Actual')
    plt.plot(X_test, label='Predicted')
    #TODO zde nejak vymyslet jinou pricelinu - jako lightweight chart
    plt.plot(y_test_ref, label='reference column - price')
    plt.plot()
    plt.legend()
    plt.savefig("res_pred_act.png")
    #plt.show()


if __name__ == "__main__":
    main()