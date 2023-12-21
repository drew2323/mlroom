import numpy as np

#https://chat.openai.com/c/7ae7057c-abb2-405f-8928-b9e78bc54619
"""""
TBD co daily data
musim si ujasnit, jak presne budou postavene sekvence v inputech
a to jak pri treninku tak predictu

1) idea byla pri predictu tam posilat to, co zrovna mam - tzn. 
napr. tick price a 5 zpatky + (u baru beru pripadne nacaty + u indikatoru posledni hodnotu)


tzn. ticky(5) 1,1,1,1,1,1
bary 0,1,1,1 (4)
ind  0,2,2,2 (4)

v rámci své maximální sekvence, pokud zde nejsou pak doplnime nulamai
tzn. kazdy má sve rozliseni a delku sekvence (5,4,4)

2) pri vectorovém treninku, potrebujeme generovat stejné, ale z hotovych dat, tzn.:
nejvyssi rozliseni definuje rytmus
tick_ind
  time       =  1,2,3,4,5,6,7
  tick_price =  2,2,2,2,2,2,2
vyssi rozliseni pak expandujeme do vyssiho rozliseni, aby odpovidalo hodnotam posilanych pri
predictu, ale se zachovanim pvuodniho rozliseni a resime pouze POSLEDNI HODNOTU
- pokud se timestampy matchuji - použijeme tuto hodnotu
- pokud hodnotu nemame (timeatamp vyssiho je vyssi) bereme prvni predchozi a
- u baru interpolujeme na hodnoty z nasledujiciho timestampu a z teto hodnoty interpolujeme
- u indikatoru bereme posledni

Jak tedy postavit highres a lowres sekvence, aby mohli byt 1:1 plneny do modelu a odpovidali realite?

Cíl je - stejný počet samplů (v tomto případě daný délkou nejvyššího rozlišení)

zjednodušeně: -  sekvencujeme nad nejvyšším rozlišením, koukneme na nižší, řízneme
v místě timestampu a vracíme co je vlevo padnuté do délky série

komplexně:
- tzn. sekvencujeme nad nejvyšším rozlišením
    - koukneme na nižší rozlišení 
         - řízneme v místě timestampu nejvyššího
         - buď máme hned timestamp, vezmeme délku sekvence(zbytek padneme) a vracíme
         - když nemáme timestamp bereme poslední(případně interpolujeme) a vezmeme délku sekvence(zbytek padneme) a vrací
         takto zpracujeme všechny nižší rozlišení a jedeme dál
         (promyslet ještě dailyBars)

Tuto analýzuj spojit s předchozí a dát do ChatGPT pro vytvoření.
POdívat se, zda by se tak nedala upravit stávající.

- s tim, ze kdyz bar neni uzavren
bere se posledni hodnota
TICK 2.50 2.52 2.53 2.52 2.54
BAR  2.50      2.52
nicmene trenink je vectorizovany, tzn bere se vzdy jedna+jedna sekvence a ta
"""


"""
 Create sequencing for all resolutions, lower resolutions are aligned with timestamp.
    Each resolution has its sequence length and number of features (keys that != time)
NUmber of samples corresponds to the highest resolutions
Output is for each resolution in shape (samples, sequence_length_of_resolution, number of feature)
"""

def create_sequences(high_res, low_res1, daily_res2):
    high_resolution_sequences = []
    low_resolution1_sequences = []
    daily_resolution2_sequences = []

    # Go through each timestamp in the high resolution data
    for timestamp in high_res['time']:
        # Find the corresponding timestamp in the low resolution data
        low_resolution1_timestamp = next(
            (t for t in low_res1['time'] if t == timestamp), None)

        # Find the corresponding timestamp in the daily resolution data
        daily_resolution2_timestamp = next(
            (t for t in daily_res2['time'] if t == timestamp), None)

        # Extract the relevant data for each resolution
        high_resolution_features = {
            'high_feature1': high_res['high_feature1'][timestamp-high_res['sequence_length']:timestamp],
            'high_feature2': high_res['high_feature2'][timestamp-high_res['sequence_length']:timestamp]
        }

        low_resolution1_features = {
            'low_feature1': low_res1['low_feature1'][low_resolution1_timestamp-low_res1['sequence_length']:low_resolution1_timestamp],
            'low_feature2': low_res1['low_feature2'][low_resolution1_timestamp-low_res1['sequence_length']:low_resolution1_timestamp]
        }

        daily_resolution2_features = {
            'close': daily_res2['close'][daily_resolution2_timestamp-daily_res2['sequence_length']:daily_resolution2_timestamp],
            'open': daily_res2['open'][daily_resolution2_timestamp-daily_res2['sequence_length']:daily_resolution2_timestamp]
        }

        # Create the sequences for each resolution
        high_resolution_sequences.append(high_resolution_features)
        low_resolution1_sequences.append(low_resolution1_features)
        daily_resolution2_sequences.append(daily_resolution2_features)

    # Convert the sequences to NumPy arrays
    high_resolution_sequences = np.array(high_resolution_sequences)
    low_resolution1_sequences = np.array(low_resolution1_sequences)
    daily_resolution2_sequences = np.array(daily_resolution2_sequences)

    return high_resolution_sequences, low_resolution1_sequences, daily_resolution2_sequences

if __name__ == '__main__':
    high_res = {
        "sequence_length": 4,
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'high_feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'high_feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1]
    }
    low_res1 = {
        "sequence_length": 2,
        'time': [3, 5, 8, 11, 14, 16],
        'low_feature1': [5, 8, 11, 14, 16, 3],
        'low_feature2': [8, 11, 14, 16, 3, 5]
    }
    daily_res2 = {
    'time': [0.3, 0.5, 0.8, 0.9, 0.95, 0.99], 
    'close': [5, 8, 11, 14, 16, 3],
    'open': [8, 11, 14, 16, 3, 5],
}
    
a,b,c = create_sequences(high_res,low_res1,daily_res2)

print(a,b,c)