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
# Updated dictionaries with two features each
high_res = {
    'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
    'high_feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'high_feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1],
}
low_res = {
    'time': [3, 5, 8, 11, 14, 16], 
    'low_feature1': [5, 8, 11, 14, 16, 3],
    'low_feature2': [8, 11, 14, 16, 3, 5],
}

daily_res = {
    'time': [0.3, 0.5, 0.8, 0.9, 0.95, 0.99], 
    'close': [5, 8, 11, 14, 16, 3],
    'open': [8, 11, 14, 16, 3, 5],
}


# Vectorized function to expand features of low_res to match high_res
def expand_low_res_features_vectorized(high_res, low_res, feature_name):
    indices = np.searchsorted(high_res["time"], low_res["time"], side='right') - 1
    expanded_values = np.zeros(len(high_res["time"]))

    for i, idx in enumerate(indices):
        expanded_values[idx] = low_res[feature_name][i]

    # Forward fill
    valid_idx = 0
    for i in range(len(expanded_values)):
        if expanded_values[i] != 0:
            valid_idx = i
            break

    for i in range(valid_idx, len(expanded_values)):
        if expanded_values[i] == 0:
            expanded_values[i] = expanded_values[i - 1]

    return expanded_values

# Expand low_feature1 and low_feature2
expanded_low_feature1 = expand_low_res_features_vectorized(high_res, low_res, 'low_feature1')
expanded_low_feature2 = expand_low_res_features_vectorized(high_res, low_res, 'low_feature2')
daily_close_expanded = expand_low_res_features_vectorized(high_res, daily_res, 'close')
print(daily_close_expanded)
#print(expanded_low_feature1, expanded_low_feature2)

