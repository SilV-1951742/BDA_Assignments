# Setup spark


```python
import os
os.environ['PYSPARK_SUBMIT_ARGS'] ="pyspark-shell"

from pyspark.sql import SparkSession
from pyspark.sql.functions import flatten
from pyspark.sql.types import (StructType, StructField, StringType, 
                                FloatType, DateType, IntegerType, ArrayType)
spark = SparkSession \
        .builder \
        .master("local[*]") \
        .config("spark.driver.memory", "15g") \
        .appName("BDA assignment") \
        .getOrCreate()
```

# Imports


```python
from typing import NamedTuple, Final, List
#from lxml import etree
import xml.etree.ElementTree as ET
from itertools import islice, chain, combinations
import argparse
import traceback
import bleach
import html
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import pr
#from pyspark.ml.feature import StopWordsRemover
import string
import random
import hashlib
import pandas as pd
import numpy as np
import hashlib
import json
import sys
from itertools import islice
```

# Constants


```python
SHINGLE_SIZE: Final = 5
SAMPLES: Final = 1000
```


```python
class comment_tuple(NamedTuple):
    id: int
    #owner_id: int
    post_type: int
    score: int
    text: str

class shingle_set(NamedTuple):
    id: int
    shingles: frozenset[tuple]

class similarity(NamedTuple):
    id_set1: int
    id_set2: int
    similarity: float
```

# Read and clean XML


```python
def set_schema():
    """
    Define the schema for the DataFrame
    """
    schema_list = []
    schema_list.append(StructField("Id", IntegerType(), True))
    #schema_list.append(StructField("PostTypeId", IntegerType(), True))
    #schema_list.append(StructField("Score", IntegerType(), True))
    schema_list.append(StructField("Body", StringType(), True))
    
    return StructType(schema_list)

def parse_post(rdd):
    results = []
    root = ET.fromstring(rdd[0])

    for elem in root.findall('row'):
        rec = []
        #print("Found row")
        assert elem.text is None, "The row wasn't empty"
        rec.append(int(elem.attrib["Id"]))
        #int(elem.attrib["OwnerUserId"]),
        #rec.append(int(elem.attrib["PostTypeId"])),
        #rec.append(int(elem.attrib["Score"])),
        rec.append(bleach.clean(elem.attrib["Body"], strip=True))
        #rec.append(elem.attrib["Body"])

        #elem.clear()
        #while elem.getprevious() is not None:
        #    del elem.getparent()[0]
        results.append(rec)

    return results
```


```python
filename = "crypto_posts.xml"
chunksize = 1024

file_rdd = spark.read.text(filename, wholetext=True).rdd
dataset = file_rdd.flatMap(parse_post)
```

# Shingling


```python
STOPWORDS: Final = stopwords.words('english')

def tokenize(text: str) -> List[str]:
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text_nop = text.split()
    text_nop = word_tokenize(text)
    filtered_words = []

    for word in text_nop:
        if word not in STOPWORDS:
            filtered_words.append(word.lower())
        
    return filtered_words

    #def create_shingle(self, input_comment: comment_tuple, shingle_size: int) -> frozenset[tuple]:
    #    tokens = self.tokenize(input_comment.text)
    #    comment_length = len(tokens)
    #    shingles =  frozenset(tuple(tokens[i:(i + shingle_size)]) for i in range(comment_length - shingle_size + 1))
    
def create_shingle(post: str, shingle_size: int) -> list[list]:
    tokens = tokenize(post)
    comment_length = len(tokens)
    shingle_set = frozenset(hash(tuple(tokens[i:(i + shingle_size - 1)])) for i in range(comment_length - shingle_size))
    shingle_list = list(shingle_set)
    shingle_list.sort()
    return shingle_list
        

def shingle_map(row):
    return (row[0], create_shingle(row[1], SHINGLE_SIZE)
)

def set_shingle_schema():
    """
    Define the schema for the DataFrame
    """
    schema_list = []
    schema_list.append(StructField("Id", IntegerType(), True))
    #schema_list.append(StructField("PostTypeId", IntegerType(), True))
    #schema_list.append(StructField("Score", IntegerType(), True))
    schema_list.append(StructField("Shingles", ArrayType(ArrayType(StringType()), True)))
    return StructType(schema_list)
```


```python
schema = set_shingle_schema()
shingle_rdd = dataset.map(shingle_map)

for elem in shingle_rdd.take(1):
    print(elem)
```

    (3, [-4290581696087936001, -665577018445718155, -628234527464423806, 510782240241182547, 4596341531489039942, 5014231163037982206])


# MinHashing


```python
# Transform posts to characteristic matrix
# Make feature set matrix
# Minhash
# Make Minhash Matrix
# LSH
import random
SIGNATURE_SIZE: Final = 108
HASH_PRIME: Final = (1 << 31) - 1
MAX_HASH: Final = (1 << 32)
    #sys.maxsize
HASH_RANGE: Final = (1<< 32)
    #sys.maxsize
SEED: Final = 193120

generator = np.random.RandomState(SEED)
salts = [generator.randint(1, MAX_HASH) for _ in range(SIGNATURE_SIZE)]
#permutations = [generator.randint(1, HASH_PRIME) for _ in range(SIGNATURE_SIZE)]

permutations = [[generator.randint(1, MAX_HASH) for _ in range(SIGNATURE_SIZE)],
                [generator.randint(0, MAX_HASH) for _ in range(SIGNATURE_SIZE)]]

print(salts)
print(permutations)
    
def min_hasher(row):
    sig = np.full((SIGNATURE_SIZE), MAX_HASH)
    for shingle in row[1]:
        for i in range(SIGNATURE_SIZE):
            a = permutations[0][i]
            b = permutations[1][i]
            hash_val = (a * shingle + b) % HASH_PRIME
            sig[i] = min(hash_val, sig[i])
    return (row[0], sig)
```

    [2098660572, 74013022, 2675907171, 2481944613, 2289028529, 846582864, 1024003631, 3457496236, 958218556, 2872479854, 1385329197, 2720560315, 2596604670, 4118717270, 3831528778, 4184700433, 130382780, 4221132295, 2492231677, 1349853675, 2674011478, 4230155413, 522369175, 1349933754, 2597169981, 1045439184, 3199517916, 3020468163, 618450142, 3282454786, 4061764145, 3477766529, 3055070885, 204747729, 834378094, 4046990333, 2141660100, 2118590568, 3537961224, 168082228, 652073352, 1284401985, 2826423363, 1325707607, 3731067674, 723716237, 3333080471, 2787841228, 2113795590, 3230829213, 1843969928, 746019385, 2309594694, 2229839123, 489336389, 779150137, 1446103079, 889348879, 3745936412, 4281307413, 2173665908, 1717438567, 23839969, 2145022294, 1052302893, 1118678081, 1295195691, 3851408124, 3928788718, 2370966349, 561561790, 1051546482, 4224882136, 4206633849, 1329396735, 1801554849, 1138195103, 3123692201, 1122989246, 1270364612, 1593839649, 1463959047, 4229718824, 1709163888, 33163415, 2667737317, 2446567315, 4184687507, 3396502067, 483813678, 2220062805, 290402151, 576718488, 2834691631, 2746963148, 1838059277, 3070124528, 885776293, 1650043432, 3582686598, 220792128, 2535197383, 1699971711, 1898865051, 3798071318, 2827338236, 435208720, 1570192010]
    [[2360615334, 515231860, 643501, 4187339057, 509908783, 486802098, 3677215786, 2921827921, 3237468090, 3427794386, 4032002656, 4093886241, 2687035725, 3077734643, 1170835918, 432938308, 2306967323, 2141534379, 2906718317, 1574298931, 4232243767, 240355278, 3647978043, 2451964705, 3514275297, 2490192023, 1350263173, 1011477999, 1728226504, 2593311831, 2348203626, 2090761939, 2237486930, 2969854286, 3729324259, 2729987854, 2824894020, 4163387809, 3551421044, 4153788825, 2068844360, 3459047484, 1942759090, 2326706854, 3182519774, 2346782325, 3000600301, 1767426993, 4219791421, 2462074449, 1606408211, 2669977725, 556556695, 54768862, 3945160854, 4094924852, 2618894079, 1399513731, 3121010025, 1078380893, 1689835941, 1565617914, 296520364, 1561541048, 447627488, 2558517354, 2863467599, 2031741865, 1917227722, 1401748806, 1919852463, 799390813, 3667330196, 3920267931, 1688006343, 1880151789, 1721222033, 1350060576, 1421179143, 3390752994, 1093982970, 176879995, 1424471741, 990809676, 3057948133, 3839456555, 3537291819, 2982473332, 1138038642, 3641814918, 3996706582, 4289385726, 3195111886, 2742126307, 2191011012, 1487626248, 4062151955, 3085559586, 3650716253, 2547517663, 3894541059, 4243427467, 2671924017, 2410756795, 2351573155, 1939820273, 1381993921, 3921890242], [2069837973, 3728365543, 3470321288, 1411551473, 2976451965, 3864841989, 2978844665, 3744852957, 3552986518, 668364357, 2395803542, 3871637363, 4196804728, 379110799, 1730082894, 1248609580, 1104288650, 2162966966, 463869164, 3779608657, 3493061710, 3831624172, 3490505061, 516033924, 1649650644, 1093784065, 2149597605, 1410190642, 1177166464, 1285722951, 261973428, 1375458813, 3736080958, 415371718, 2833666829, 2672402390, 91406413, 2151211436, 423049810, 1607572258, 4266683143, 1285531306, 757827060, 343450530, 25323917, 2352946753, 1150347800, 2057611232, 1674351078, 2076247240, 146794922, 106747079, 4283727945, 3483916802, 1387832390, 1809685343, 4084122288, 13478209, 3941993887, 480095244, 869413593, 394705573, 907594834, 4246996269, 1325703698, 2968061611, 3085116452, 3984611551, 1987496472, 260011613, 1420349554, 2644598353, 2851332728, 2253089369, 3941313582, 339466831, 3592294033, 1853322040, 1419670630, 201116580, 1675447312, 2749758996, 4246665421, 3985001804, 2098346367, 1627591860, 2882913549, 3038258133, 1725294169, 989935357, 1762334801, 3981094314, 1432056049, 2569351693, 3653989293, 3877767251, 2626422348, 860587705, 306276007, 847903563, 4072136872, 3399657919, 11231594, 1411385303, 3706797702, 2610923441, 894568305, 3240125565]]



```python
a = permutations[0][1]
b = permutations[1][1]

hash_val = (a * hash(tuple(["text", "word", "shingle", "advanced"]))+ b) % HASH_PRIME
print(hash_val)

hash_val = (a * hash(tuple(["text", "word", "shingle", "advanced"]))+ b) % HASH_PRIME
print(hash_val)
```

    912218934
    912218934



```python
hash_rdd = shingle_rdd.map(min_hasher)

for elem in hash_rdd.take(3):
    print(elem)
```

    (3, array([ 208013561,  675384385,   48336623,  119638836,  349973794,
             55802641,  625474568,   44931900,  578865653,   19757788,
            898408950, 1032203146,  266430331,  193001155,  257639267,
            231732010,   13697903,  427297710,  259256966,  347411380,
            589433502,  562075005,    8030302,  459097122,  672363302,
            649588980,  485561913,  571493232,  117436184,  321519854,
             63179342,  801987483,  176130970,  576242069,   21406281,
            339084354,  431504037,  130967897,   62653543,  458730357,
            392972913,  850558799,  316009115,  369021481,  195190506,
            356619147,  855430275,  125104144,  261887145,  658860329,
            228481084,  395480755,  317592276,  251123471,   53990565,
            136885178,  421422023,  479285663,  278006285,  117059182,
            967430912,  388011285,  135435827,   84234646,  303146999,
             27239454,  905026519,  555477674,  248248992,  516533216,
            711710801,  457891682,   42161973,  277883322,  705545689,
            274898163,  186279381,  123938354, 1071339748,  269574440,
            330924802,  245035364,  368532056,   30844009,  354371493,
             20590412,  113922956,   50926457,  148996506,  403011962,
            572854049,  531064244,    2634768,  226318973,  135307312,
            130433897,  531440089,  489335729,    9451799,  630487416,
            740952341,  113918297,  180441457,   85093760,   35430660,
             67790934,  870273376,   82067432]))
    (6, array([20393214,  8991358, 16576929,  8246043,   585142,  2231220,
           17384137,  2954723, 27648367,  5527275, 11736564,  5837612,
            8523387, 15419919, 15303383,  4203541,  3463515,  2739410,
            2172002, 11562530, 40464672, 10163574,  9846580,  8298496,
            1531375,   402029,  6095081,  2960065, 12657889,   646551,
           11957403,  4085801, 24354623,   301224, 25237723, 18048329,
           23177042,  2253394, 13921987,  3276844,  1018084, 35285167,
           15202611, 37716246,  1660554,  6531923,  6637818, 17175100,
             286295, 11232074,   464294, 30878579,  5729468,  4027928,
           10194816,   650668, 13777609, 14305245,  2391387, 15845012,
            7720187,   994831,  2201809,  2172848, 12523331,  8323930,
             479120,  8634871, 26582786,   718949, 30698307,  1197453,
            5578085,  3048945,   665852,  6179754, 19921886,  5003594,
            7139808,  2979835, 12327208,   182574,  2907721,  4602133,
           23459578,  4599876,  4431044,  2176171, 11948138, 11430298,
            2245420, 27103253,  5385565,  6800816,  5539307,   534782,
           17705999,   355156,  7354129,  2433955,  8585645,  6640036,
            3581504,  9260425,   396028,   466567, 14814175, 21341196]))
    (7, array([236715974,  34903183,  26809022,   6971969, 256319513,  89118323,
            43040765,   7176289,  51050674, 115083065,  56376932,   3990470,
           109168934,  13719395, 242304882,  49490625, 189318402, 133481255,
            43906669, 140478346,  80498263,  82413632, 117881403,  24271772,
           183176235,  66714328, 403055585,  14319604,  29111719, 262331716,
            39668542,  10696412, 324099926,  18953049,    921308,   6770342,
            21361632, 178886243,  62496248, 113010578, 107320370,  12236046,
            16644591,  25456779, 213725804, 184247461,  37159055,  51757705,
           299019386,  75372254,  12498892,    509830, 178946374, 217608646,
            41178813,   6492795,  38918166, 219804540,  37666243,   2217108,
           201404360,  46944959,  60691505,  32706838, 205587760,  29249158,
            67843676, 218182035,  29943749, 191656604, 170479689,  10368042,
              472188,  73162250,  41684484, 185425960, 125002587, 108613241,
            91481379, 101646228,  73040476,  41523955, 114741325,  63522598,
            73920008,   4158023,  75316875,  40926489, 119046954,  85648685,
            36054048, 147991888,  73263083,  27031789,   8419590,  94421593,
           187832049,  51396667, 282833743,  56208566,  35633841,  43358002,
            15806685,  87266503, 100042963, 215103302,   8594384, 130425793]))


# LSH


```python
BANDS: Final = 18
ROWS: Final = 6
THRESHOLD: Final = (1/BANDS) ** (1/ROWS)
print(f"Bands: {BANDS}, rows {ROWS}, threshold {THRESHOLD}")

def hash_func(row):
    sum = 0
    for e in row[1][0]:
        sum += e
    return (row[0], (int.from_bytes(hashlib.md5(str(sum).encode()).digest()[:4], byteorder="big"), row[1][1]))
```

    Bands: 18, rows 6, threshold 0.6177146705271326



```python
# returns (doc, band, hash)
hash_band_rdd = hash_rdd.flatMap(lambda x: [[(x[0], i % BANDS), hash] for i, hash in enumerate(x[1])]).groupByKey().cache()

for elem in hash_band_rdd.take(5):
    print(elem)
```

    ((281, 10), <pyspark.resultiterable.ResultIterable object at 0x7fb041024b20>)
    ((289, 15), <pyspark.resultiterable.ResultIterable object at 0x7fb041024310>)
    ((302, 2), <pyspark.resultiterable.ResultIterable object at 0x7fb040f6f040>)
    ((331, 3), <pyspark.resultiterable.ResultIterable object at 0x7fb041024070>)
    ((359, 16), <pyspark.resultiterable.ResultIterable object at 0x7fb040f6f0d0>)



```python
hash_bands_grouped_rdd = hash_band_rdd.map(lambda x: [x[0][1], (x[1], x[0][0])])
```


```python
band_hashed = hash_bands_grouped_rdd.map(hash_func).map(lambda x: [(x[0], x[1][0]), x[1][1]]).groupByKey().filter(lambda x: (len(x[1]) > 1 and len(x[1]) < 50))

for elem in band_hashed.take(10):
    print(elem[0])
    for b in elem[1]:
        print(b)
```

    (17, 3620383786)
    68068
    44601
    (12, 2015369168)
    8109
    75410
    (2, 436572783)
    9773
    9772
    (1, 353161229)
    10360
    10361
    (10, 2402892704)
    10897
    10896
    (3, 3374985088)
    11483
    11484
    (3, 2852162333)
    32382
    31090
    (13, 1963045978)
    59977
    50805
    (6, 480059632)
    87377
    87382
    (12, 2999566503)
    1209
    1210



```python
candidates = band_hashed.map(lambda x: (tuple(x[1]), 1)).reduceByKey(lambda a, b: ((float(a + b) / 10.0))).cache()

for elem in candidates.take(100):
    print(elem)
```

    ((68068, 44601), 1)
    ((8109, 75410), 1)
    ((9773, 9772), 1)
    ((10360, 10361), 0.11111120000000001)
    ((10897, 10896), 0.11111200000000002)
    ((11483, 11484), 0.11111112000000001)
    ((32382, 31090), 0.11111200000000002)
    ((59977, 50805), 0.11111111199999998)
    ((87377, 87382), 0.11200000000000002)
    ((1209, 1210), 0.111111111112)
    ((1591, 1592), 0.11111111119999999)
    ((5354, 41752), 1)
    ((8394, 8393), 0.11111111111111201)
    ((9721, 806), 1)
    ((9833, 9834), 0.11111200000000002)
    ((9834, 9833), 0.11111111119999999)
    ((9849, 9848), 0.11111120000000001)
    ((10185, 10186), 1)
    ((10622, 10621), 0.111111111112)
    ((10896, 10897), 0.11111111119999999)
    ((10969, 10970), 0.11111111112000001)
    ((11094, 11093), 0.11111120000000001)
    ((12010, 12011), 0.11111111119999999)
    ((12013, 12012), 0.11111111199999998)
    ((12732, 12733), 1)
    ((26502, 84119), 1)
    ((30136, 47862), 1)
    ((34136, 10718), 0.11111111199999998)
    ((37318, 37300), 1)
    ((44290, 68294), 0.2)
    ((50805, 59977), 0.11111120000000001)
    ((64096, 64053), 0.11200000000000002)
    ((72107, 66307), 1)
    ((72388, 72389), 0.11111120000000001)
    ((84227, 84228), 0.11111111199999998)
    ((91844, 75489), 0.12)
    ((3410, 3409), 0.11112)
    ((10621, 10622), 0.11120000000000001)
    ((10717, 34135), 0.11111111199999998)
    ((31090, 32382), 0.12)
    ((1243, 1244), 0.11111111119999999)
    ((9847, 9846), 0.11111200000000002)
    ((20787, 20789), 0.2)
    ((3416, 3415), 0.11120000000000001)
    ((42254, 41878), 0.11120000000000001)
    ((72389, 72388), 0.11111111199999998)
    ((216, 215), 0.11111111111112)
    ((218, 217), 0.11111111112000001)
    ((1590, 1589), 0.11111200000000002)
    ((10954, 10953), 0.11111111119999999)
    ((11093, 11094), 0.11111111199999998)
    ((12516, 12515), 0.11111120000000001)
    ((13212, 13149), 0.11111112000000001)
    ((15054, 15053), 0.11120000000000001)
    ((26874, 26234), 0.11111200000000002)
    ((27645, 27646), 0.11111120000000001)
    ((32601, 32510), 0.2)
    ((34135, 10717), 0.11111120000000001)
    ((48488, 48485), 0.11111112000000001)
    ((9800, 9801), 0.12)
    ((48485, 48488), 0.11111112000000001)
    ((41230, 41249), 0.2)
    ((12009, 12008), 0.11112)
    ((75461, 75473), 1)
    ((81089, 32608), 0.2)
    ((84229, 84230), 0.11200000000000002)
    ((6, 1282), 1)
    ((10892, 10893), 0.11111200000000002)
    ((10894, 3543), 0.11111111111112)
    ((11148, 11147), 0.11111111119999999)
    ((22641, 22622), 0.11111111119999999)
    ((26234, 26874), 0.11111111119999999)
    ((40036, 42051), 1)
    ((60978, 87221), 1)
    ((62252, 62218), 0.12)
    ((3409, 3410), 0.11111111112000001)
    ((10943, 10942), 0.11200000000000002)
    ((30271, 10785), 0.2)
    ((1210, 1209), 0.11120000000000001)
    ((8878, 3213), 0.2)
    ((10718, 34136), 0.11111120000000001)
    ((10970, 10969), 0.11112)
    ((12008, 12009), 0.11111111112000001)
    ((12012, 12013), 0.11111120000000001)
    ((64053, 64096), 0.2)
    ((10784, 30270), 0.12)
    ((84230, 84229), 0.1111111111112)
    ((10840, 10841), 0.11111120000000001)
    ((217, 218), 0.11112)
    ((9837, 9836), 0.111111111112)
    ((9841, 9840), 0.11111120000000001)
    ((41878, 42254), 0.111111111112)
    ((55767, 55307), 1)
    ((33509, 31490), 1)
    ((84228, 84227), 0.11111120000000001)
    ((13149, 13212), 0.11111112000000001)
    ((9840, 9841), 0.11111111199999998)
    ((12015, 12014), 0.11111111112000001)
    ((16008, 85997), 1)
    ((31700, 31574), 1)



```python
cand = candidates.collect()
shingle_dict = shingle_rdd.collectAsMap()
```


```python
def calc_jaccard(list1, list2):
    return len(set(list1).intersection(list2)) / len(set(list1).union(list2))

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
```


```python
import itertools

for elem in cand:
    #print(f"Candidates: {elem[0][0]}, {elem[0][1]}; similarity: ", end="")
    #print(f"{calc_jaccard(shingle_dict.get(elem[0][0]), shingle_dict.get(elem[0][1]))}")
    for comb in itertools.combinations(elem[0], 2):
        print(f"Candidates: {comb[0]}, {comb[1]}; similarity: ", end="")
        print(f"{calc_jaccard(shingle_dict.get(comb[0]), shingle_dict.get(comb[1]))}")
```

    Candidates: 68068, 44601; similarity: 0.0
    Candidates: 8109, 75410; similarity: 0.0
    Candidates: 9773, 9772; similarity: 0.3333333333333333
    Candidates: 10360, 10361; similarity: 1.0
    Candidates: 10897, 10896; similarity: 1.0
    Candidates: 11483, 11484; similarity: 1.0
    Candidates: 32382, 31090; similarity: 0.927710843373494
    Candidates: 59977, 50805; similarity: 1.0
    Candidates: 87377, 87382; similarity: 0.8686868686868687
    Candidates: 1209, 1210; similarity: 1.0
    Candidates: 1591, 1592; similarity: 1.0
    Candidates: 5354, 41752; similarity: 0.0
    Candidates: 8394, 8393; similarity: 1.0
    Candidates: 9721, 806; similarity: 0.0
    Candidates: 9833, 9834; similarity: 1.0
    Candidates: 9834, 9833; similarity: 1.0
    Candidates: 9849, 9848; similarity: 1.0
    Candidates: 10185, 10186; similarity: 0.4782608695652174
    Candidates: 10622, 10621; similarity: 1.0
    Candidates: 10896, 10897; similarity: 1.0
    Candidates: 10969, 10970; similarity: 1.0
    Candidates: 11094, 11093; similarity: 1.0
    Candidates: 12010, 12011; similarity: 1.0
    Candidates: 12013, 12012; similarity: 1.0
    Candidates: 12732, 12733; similarity: 0.5
    Candidates: 26502, 84119; similarity: 0.0
    Candidates: 30136, 47862; similarity: 0.0
    Candidates: 34136, 10718; similarity: 1.0
    Candidates: 37318, 37300; similarity: 0.620253164556962
    Candidates: 44290, 68294; similarity: 0.6363636363636364
    Candidates: 50805, 59977; similarity: 1.0
    Candidates: 64096, 64053; similarity: 0.8868894601542416
    Candidates: 72107, 66307; similarity: 0.0
    Candidates: 72388, 72389; similarity: 1.0
    Candidates: 84227, 84228; similarity: 1.0
    Candidates: 91844, 75489; similarity: 0.729381443298969
    Candidates: 3410, 3409; similarity: 1.0
    Candidates: 10621, 10622; similarity: 1.0
    Candidates: 10717, 34135; similarity: 1.0
    Candidates: 31090, 32382; similarity: 0.927710843373494
    Candidates: 1243, 1244; similarity: 1.0
    Candidates: 9847, 9846; similarity: 1.0
    Candidates: 20787, 20789; similarity: 0.7755102040816326
    Candidates: 3416, 3415; similarity: 0.7692307692307693
    Candidates: 42254, 41878; similarity: 1.0
    Candidates: 72389, 72388; similarity: 1.0
    Candidates: 216, 215; similarity: 1.0
    Candidates: 218, 217; similarity: 1.0
    Candidates: 1590, 1589; similarity: 1.0
    Candidates: 10954, 10953; similarity: 1.0
    Candidates: 11093, 11094; similarity: 1.0
    Candidates: 12516, 12515; similarity: 1.0
    Candidates: 13212, 13149; similarity: 1.0
    Candidates: 15054, 15053; similarity: 0.9130434782608695
    Candidates: 26874, 26234; similarity: 1.0
    Candidates: 27645, 27646; similarity: 1.0
    Candidates: 32601, 32510; similarity: 0.7411764705882353
    Candidates: 34135, 10717; similarity: 1.0
    Candidates: 48488, 48485; similarity: 1.0
    Candidates: 9800, 9801; similarity: 0.8709677419354839
    Candidates: 48485, 48488; similarity: 1.0
    Candidates: 41230, 41249; similarity: 0.7104072398190046
    Candidates: 12009, 12008; similarity: 1.0
    Candidates: 75461, 75473; similarity: 0.319047619047619
    Candidates: 81089, 32608; similarity: 0.6941176470588235
    Candidates: 84229, 84230; similarity: 1.0
    Candidates: 6, 1282; similarity: 0.6976744186046512
    Candidates: 10892, 10893; similarity: 1.0
    Candidates: 10894, 3543; similarity: 1.0
    Candidates: 11148, 11147; similarity: 1.0
    Candidates: 22641, 22622; similarity: 1.0
    Candidates: 26234, 26874; similarity: 1.0
    Candidates: 40036, 42051; similarity: 0.6732594936708861
    Candidates: 60978, 87221; similarity: 0.0
    Candidates: 62252, 62218; similarity: 0.75
    Candidates: 3409, 3410; similarity: 1.0
    Candidates: 10943, 10942; similarity: 1.0
    Candidates: 30271, 10785; similarity: 0.8571428571428571
    Candidates: 1210, 1209; similarity: 1.0
    Candidates: 8878, 3213; similarity: 0.5357142857142857
    Candidates: 10718, 34136; similarity: 1.0
    Candidates: 10970, 10969; similarity: 1.0
    Candidates: 12008, 12009; similarity: 1.0
    Candidates: 12012, 12013; similarity: 1.0
    Candidates: 64053, 64096; similarity: 0.8868894601542416
    Candidates: 10784, 30270; similarity: 0.967741935483871
    Candidates: 84230, 84229; similarity: 1.0
    Candidates: 10840, 10841; similarity: 1.0
    Candidates: 217, 218; similarity: 1.0
    Candidates: 9837, 9836; similarity: 1.0
    Candidates: 9841, 9840; similarity: 1.0
    Candidates: 41878, 42254; similarity: 1.0
    Candidates: 55767, 55307; similarity: 0.5797101449275363
    Candidates: 33509, 31490; similarity: 0.0
    Candidates: 84228, 84227; similarity: 1.0
    Candidates: 13149, 13212; similarity: 1.0
    Candidates: 9840, 9841; similarity: 1.0
    Candidates: 12015, 12014; similarity: 1.0
    Candidates: 16008, 85997; similarity: 0.0
    Candidates: 31700, 31574; similarity: 0.3333333333333333
    Candidates: 84299, 84298; similarity: 0.7173333333333334
    Candidates: 60504, 81065; similarity: 0.0
    Candidates: 9848, 9849; similarity: 1.0
    Candidates: 11484, 11483; similarity: 1.0
    Candidates: 12014, 12015; similarity: 1.0
    Candidates: 29210, 1572; similarity: 0.0
    Candidates: 12277, 12278; similarity: 0.5625
    Candidates: 84298, 84299; similarity: 0.7173333333333334
    Candidates: 9884, 9885; similarity: 1.0
    Candidates: 10361, 10360; similarity: 1.0
    Candidates: 81007, 15927; similarity: 0.21739130434782608
    Candidates: 1589, 1590; similarity: 1.0
    Candidates: 32306, 31669; similarity: 0.5636363636363636
    Candidates: 53359, 88398; similarity: 0.0
    Candidates: 5430, 1127; similarity: 0.0
    Candidates: 15053, 15054; similarity: 0.9130434782608695
    Candidates: 32240, 91687; similarity: 0.0
    Candidates: 58711, 58712; similarity: 0.8367346938775511
    Candidates: 58712, 58711; similarity: 0.8367346938775511
    Candidates: 72638, 16349; similarity: 0.0
    Candidates: 9885, 9884; similarity: 1.0
    Candidates: 13103, 10772; similarity: 0.7682926829268293
    Candidates: 30270, 10784; similarity: 0.967741935483871
    Candidates: 47579, 47531; similarity: 0.7613636363636364
    Candidates: 59812, 34969; similarity: 0.4716981132075472
    Candidates: 75238, 34682; similarity: 0.0
    Candidates: 10942, 10943; similarity: 1.0
    Candidates: 31659, 16259; similarity: 0.0
    Candidates: 57772, 20392; similarity: 0.0
    Candidates: 67859, 18684; similarity: 0.0
    Candidates: 80774, 9248; similarity: 0.0
    Candidates: 91814, 91828; similarity: 0.75
    Candidates: 33683, 33696; similarity: 0.5348837209302325
    Candidates: 1207, 1208; similarity: 0.8076923076923077
    Candidates: 33042, 33043; similarity: 0.8823529411764706
    Candidates: 83709, 63932; similarity: 0.0
    Candidates: 11147, 11148; similarity: 1.0
    Candidates: 68225, 19427; similarity: 0.0
    Candidates: 12515, 12516; similarity: 1.0
    Candidates: 10841, 10840; similarity: 1.0
    Candidates: 9846, 9847; similarity: 1.0
    Candidates: 10966, 10965; similarity: 0.8
    Candidates: 12148, 12149; similarity: 0.43902439024390244
    Candidates: 31798, 16327; similarity: 0.0
    Candidates: 87314, 26726; similarity: 0.0
    Candidates: 10965, 10966; similarity: 0.8
    Candidates: 10893, 10892; similarity: 1.0
    Candidates: 1592, 1591; similarity: 1.0
    Candidates: 8111, 27181; similarity: 0.0
    Candidates: 8393, 8394; similarity: 1.0
    Candidates: 18715, 76881; similarity: 0.0
    Candidates: 30590, 29760; similarity: 0.0
    Candidates: 33043, 33042; similarity: 0.8823529411764706
    Candidates: 87382, 87377; similarity: 0.8686868686868687
    Candidates: 8695, 87952; similarity: 0.0
    Candidates: 10620, 10619; similarity: 1.0
    Candidates: 51348, 51349; similarity: 0.42857142857142855
    Candidates: 53961, 5237; similarity: 0.0
    Candidates: 11796, 11795; similarity: 0.7391304347826086
    Candidates: 27646, 27645; similarity: 1.0
    Candidates: 234, 233; similarity: 0.8571428571428571
    Candidates: 54143, 71718; similarity: 0.0
    Candidates: 55007, 43412; similarity: 0.0
    Candidates: 11607, 11606; similarity: 0.8571428571428571
    Candidates: 1227, 1228; similarity: 0.5
    Candidates: 1244, 1243; similarity: 1.0
    Candidates: 41249, 41230; similarity: 0.7104072398190046
    Candidates: 58186, 11039; similarity: 0.0
    Candidates: 67120, 84423; similarity: 0.0
    Candidates: 9836, 9837; similarity: 1.0
    Candidates: 31561, 31683; similarity: 0.5507246376811594
    Candidates: 62218, 62252; similarity: 0.75
    Candidates: 87121, 60376; similarity: 0.0
    Candidates: 31990, 51836; similarity: 0.0
    Candidates: 80865, 58922; similarity: 0.0
    Candidates: 8869, 8871; similarity: 0.5
    Candidates: 62490, 90163; similarity: 0.0
    Candidates: 52176, 48799; similarity: 0.0
    Candidates: 10953, 10954; similarity: 1.0
    Candidates: 77634, 67725; similarity: 0.6533333333333333
    Candidates: 150, 719; similarity: 0.5071942446043165
    Candidates: 5626, 30799; similarity: 0.0
    Candidates: 9801, 9800; similarity: 0.8709677419354839
    Candidates: 10619, 10620; similarity: 1.0
    Candidates: 1208, 1207; similarity: 0.8076923076923077
    Candidates: 16481, 5525; similarity: 0.0
    Candidates: 81085, 16039; similarity: 0.5846153846153846
    Candidates: 22622, 22641; similarity: 1.0
    Candidates: 3415, 3416; similarity: 0.7692307692307693
    Candidates: 3213, 8869; similarity: 0.6666666666666666
    Candidates: 3213, 8878; similarity: 0.5357142857142857
    Candidates: 8869, 8878; similarity: 0.5769230769230769
    Candidates: 9831, 9832; similarity: 0.5217391304347826
    Candidates: 11927, 75489; similarity: 0.6302895322939867
    Candidates: 11927, 91844; similarity: 0.5305486284289277
    Candidates: 75489, 91844; similarity: 0.729381443298969
    Candidates: 17617, 16615; similarity: 0.3856041131105398
    Candidates: 84291, 59174; similarity: 0.0
    Candidates: 51991, 87303; similarity: 0.0
    Candidates: 45252, 81630; similarity: 0.0
    Candidates: 12011, 12010; similarity: 1.0
    Candidates: 26514, 26794; similarity: 0.5135135135135135
    Candidates: 34476, 3945; similarity: 0.0
    Candidates: 10772, 13103; similarity: 0.7682926829268293
    Candidates: 93607, 10041; similarity: 0.0
    Candidates: 2170, 44474; similarity: 0.6
    Candidates: 8030, 16216; similarity: 0.0
    Candidates: 48639, 26526; similarity: 0.0
    Candidates: 47531, 47579; similarity: 0.7613636363636364
    Candidates: 64688, 64684; similarity: 0.5413533834586466
    Candidates: 215, 216; similarity: 1.0
    Candidates: 233, 234; similarity: 0.8571428571428571
    Candidates: 35378, 33628; similarity: 0.6333333333333333
    Candidates: 68839, 68800; similarity: 0.6530612244897959
    Candidates: 81357, 93546; similarity: 0.0
    Candidates: 88738, 23994; similarity: 0.3877551020408163
    Candidates: 16306, 30183; similarity: 0.0
    Candidates: 3543, 10894; similarity: 1.0
    Candidates: 5197, 1560; similarity: 0.6744186046511628
    Candidates: 25693, 26418; similarity: 0.0
    Candidates: 16039, 81085; similarity: 0.5846153846153846
    Candidates: 346, 19120; similarity: 0.0
    Candidates: 11606, 11607; similarity: 0.8571428571428571
    Candidates: 18455, 6286; similarity: 0.0
    Candidates: 20636, 20667; similarity: 0.5543478260869565
    Candidates: 44035, 60457; similarity: 0.0
    Candidates: 88553, 86202; similarity: 0.0
    Candidates: 20578, 45381; similarity: 0.0
    Candidates: 25237, 59155; similarity: 0.0
    Candidates: 10785, 30271; similarity: 0.8571428571428571
    Candidates: 9830, 9829; similarity: 0.5476190476190477
    Candidates: 22566, 42749; similarity: 0.0
    Candidates: 74587, 10629; similarity: 0.0
    Candidates: 15298, 15309; similarity: 0.6582278481012658
    Candidates: 62401, 52837; similarity: 0.0
    Candidates: 68086, 15362; similarity: 0.0
    Candidates: 24441, 24459; similarity: 0.6744186046511628
    Candidates: 9844, 9845; similarity: 0.5238095238095238
    Candidates: 59213, 6698; similarity: 0.0
    Candidates: 67066, 33475; similarity: 0.5465116279069767
    Candidates: 66525, 19433; similarity: 0.0
    Candidates: 68698, 14322; similarity: 0.0
    Candidates: 30058, 83806; similarity: 0.4666666666666667
    Candidates: 41904, 62191; similarity: 0.0
    Candidates: 1228, 1227; similarity: 0.5
    Candidates: 91828, 91814; similarity: 0.75
    Candidates: 80873, 67540; similarity: 0.5384615384615384
    Candidates: 61766, 31764; similarity: 0.0
    Candidates: 72127, 9576; similarity: 0.0
    Candidates: 80234, 51388; similarity: 0.0
    Candidates: 67725, 77634; similarity: 0.6533333333333333
    Candidates: 62385, 54141; similarity: 0.0
    Candidates: 1282, 6; similarity: 0.6976744186046512
    Candidates: 84472, 68130; similarity: 0.0
    Candidates: 85832, 70383; similarity: 0.0
    Candidates: 9548, 9550; similarity: 0.5454545454545454


# Exit Spark


```python
spark.stop()
```


```python

```
