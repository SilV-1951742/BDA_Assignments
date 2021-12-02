# Setup spark


```python
import os
os.environ['PYSPARK_SUBMIT_ARGS'] ="--conf spark.driver.memory=4g  pyspark-shell"

from pyspark.sql import SparkSession
from pyspark.sql.functions import flatten
from pyspark.sql.types import (StructType, StructField, StringType, 
                                FloatType, DateType, IntegerType, ArrayType)
spark = SparkSession \
        .builder \
        .master("local[*]") \
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

for elem in shingle_rdd.take(5):
    print(elem)
```

    (3, [-4290581696087936001, -665577018445718155, -628234527464423806, 510782240241182547, 4596341531489039942, 5014231163037982206])
    (6, [-9111936581814511609, -9095568451866840300, -9047861504496125942, -8988003250329032313, -8975985032446626129, -8735463266873975758, -8639630414950950011, -8535597839904948788, -8347401456234809095, -8325156051403803781, -8254861877436285502, -8232255744862697296, -8109278151139910983, -8099290211966067582, -8046941216478882179, -7947504582858383715, -7906491602762591470, -7801512941547215210, -7770720765893075552, -7728752191107116541, -7606172323661403454, -7557120737344180788, -7460563734288286386, -7362114286777445087, -7255100564934241087, -7221198766623982960, -7214008681594613905, -7202607934367035857, -7001762764223832935, -6978302683948703967, -6761041461593885292, -6758297973369292159, -6600858731646495027, -6566174900866955359, -6477373517399758247, -6439784059086361183, -6422649711487709871, -6330311677399841507, -6203514089950190511, -6190909300851096985, -6160811422621071947, -6101521455504449048, -6096806981602513892, -5968616583528144792, -5717182981869536543, -5619278094090755795, -5597038509610004191, -5354957881777243601, -5340801691079672728, -5323133901844398585, -5268450436093796927, -5256349971872002128, -5253404359751358890, -5058334457528168657, -5036451909714095968, -5017096221717873231, -4996842885001296168, -4209666577718750372, -4161698426022706738, -4140919240499243989, -3895994402745694684, -3864276496372623232, -3764939089842660108, -3721909064733890275, -3629382631631392348, -3618187717848716279, -3566184419628380232, -3558843882097360577, -3555962394546418350, -3365384667722603860, -3289293964028939392, -3261273208517068943, -3183412675303633914, -3178637845837077868, -3172684901455680432, -3156022459164911891, -3033355021663594257, -2947427449650796950, -2867806749702507533, -2750369654979702293, -2749041759011732480, -2586525931586103417, -2580390088610178594, -2415738680324311172, -2334365447161206160, -2301936577275730602, -2235420307918456931, -2081024595534950535, -2058637429681177053, -1979464238285472022, -1951535786178627602, -1906871741777200049, -1889571079965052073, -1888664778616937501, -1860365652366879639, -1636061310253990800, -1594256473393210059, -1594242774367943411, -1478768331810583950, -1321383922758336466, -1268491526016524798, -1135403238123788825, -1054693019356998781, -1049211263383965699, -807236973860564235, -781652058374472900, -733919868867741715, -717553150570410512, -642399171674814949, -512563363883594712, -448621354841806613, -394828796262330964, -269069551026053182, -165770384550916630, -137027429199312633, 38423205104953980, 46841549911210932, 101687591281564249, 199783265198851669, 213907072137370137, 235527100699843662, 350595117196496522, 359239481902058531, 386600872155575769, 414188787597057768, 418751585283620297, 505987532275638697, 622251688973957461, 751204120442971607, 779537292398606377, 915599857771015799, 936337351552365878, 955925608684809758, 968131768062161801, 1033295506737258242, 1054705731981898492, 1085847456320931214, 1199346315528090552, 1501430109696692595, 1579365313133921299, 1817102846322722036, 1863465945536143834, 1919293289402213074, 1961070152671837828, 1964116781590927509, 1974368270345022866, 2014219382412416437, 2078665443292592609, 2386248518910093246, 2397634416646573047, 2431828769506912716, 2474140144058457894, 2804274201848685764, 2852162926409059617, 2896958883217735557, 2899474065926467701, 2907181326398931914, 2980195243457785815, 3073992164446940817, 3196662139362347849, 3301646534089567750, 3315980709329507640, 3347891155642534692, 3378752292874731784, 3385928171132386459, 3427751575569055191, 3443166132478144396, 3462321026426591922, 3515719435536468386, 3524124981233465150, 3710812223879195656, 3836696100615621688, 3942911126419151584, 4086953551474009533, 4137554336010450607, 4147145931463536497, 4166986453845097521, 4191081567633528839, 4206035957699628720, 4347862218114080303, 4382298498670963268, 4408784918568941143, 4456883662518580652, 4457981360374473547, 4493523518292790305, 4555139512638035888, 4830401628001218349, 4846425852965312328, 5010783557842511059, 5034059151056530035, 5082163673145319649, 5117622692940540848, 5163143245772666343, 5215137380937673427, 5256951943232681796, 5314993136214863554, 5340288860485321232, 5367832678082843169, 5421110509798029392, 5513442102227737630, 5583523700972577481, 5611592047648673542, 5615140824602583003, 5651577387668512571, 5772020442018989763, 5871306226254283908, 5943243148800766281, 5949580742485720069, 5957825923347248900, 6169231893575154152, 6195437543291078360, 6362136009023398412, 6410992693953140318, 6488691156000880252, 6787670459166505904, 6828136684627329572, 6892445384913252646, 6970896809856557659, 6978088582877874601, 7093823994063805691, 7110603470865596600, 7143037007995940680, 7290834176498943526, 7292517896846804472, 7349614288985117299, 7392975081370084175, 7395448857920639115, 7396691339412857831, 7400626111266533430, 7404957921644850694, 7528602656026517113, 7545947783812006415, 7613500219789237569, 7655727858627836199, 7833665208529686608, 7947754436072686428, 8253309646249699534, 8264195667097139330, 8265990193837425089, 8275480582911713830, 8291013266172212684, 8309630278195447362, 8464345112711036515, 8535140791839232812, 8541110025690698068, 8656955575919301692, 8683981564447792231, 8705785246066348223, 8722798376567450464, 8791408579381235258, 8801388214480111278, 9100252360244295033, 9160698817963149405])
    (7, [-8855011894757223282, -7249461243601619313, -5288515085714580953, -4573871172805811642, -4524690136648079826, -3946023804224164790, -2705835220504125627, -1102163880003267999, -1081416374413349290, -787403192847218437, -739805109060580259, 539182587366915419, 617588994212417462, 2546440015100811445, 2666763638188008329, 2745189706476828669, 4101254100039029643, 4103572656080411437, 6878535434921831802, 8018833555603547317, 8348612362122056509, 8538443791451994279])
    (8, [-8312328454317395995, -3204625677901808079, -1327103818569786488])
    (9, [-8644146139282421307, -8402874006050735124, -8057704747963587570, -7749089698390289486, -7436171872503093785, -7260232483454419967, -7189411033556349303, -6808873927558965682, -6650455759697711235, -6474894697977961689, -6224221115390453135, -5742578660956142546, -5527971864271605780, -5025838338018727184, -4650155051957369717, -4599128468916653633, -4518357284660259407, -4305529774182230044, -3877513929388724050, -3376806562350058678, -2938875090112337389, -2893943735064025395, -2799126604354219596, -2686740655744802916, -1956835829439139921, -1210159275216141737, -622651483394116675, -284142132535007330, 238316651113137607, 432020083281652852, 914921587071487399, 1032844636375601318, 1454399065914734831, 1501054851843483138, 1586037246155569282, 2097941356583440684, 2618388045869901982, 2742923361208749482, 2827329140784327185, 3727682181854606629, 4360762874861810010, 4368787198587197336, 4569877551316223517, 4836361305083769478, 4981493034131687130, 4993944674953681411, 5929069226825198806, 6246312451035745645, 6266023380609598772, 6955377949513621290, 7184952115764409768, 7598725991643589150, 7631347759324128596, 7735185877639568604])



```python
#df_shingle = shingle_rdd.toDF(schema)
#df_shingle.show()
```


```python
print(hash(tuple(['step', 'hrefhttpenwikipediaorgwikidataencryptionstandarddes', 'algorithma', 'one'])))
print(hash(tuple(['why', 'use', 'permutation', 'table'])))
print(hash(tuple(['permutation', 'table', 'first', 'step'])))
print(hash(tuple(['table', 'first', 'step', 'hrefhttpenwikipediaorgwikidataencryptionstandarddes'])))
print(hash(tuple(['use', 'permutation', 'table', 'first'])))
print(hash(tuple(['first', 'step', 'hrefhttpenwikipediaorgwikidataencryptionstandarddes', 'algorithma'])))
```

    -628234527464423806
    4596341531489039942
    510782240241182547
    -665577018445718155
    5014231163037982206
    -4290581696087936001


# MinHashing


```python
# Transform posts to characteristic matrix
# Make feature set matrix
# Minhash
# Make Minhash Matrix
# LSH
import random
SIGNATURE_SIZE: Final = 45
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

    [2098660572, 74013022, 2675907171, 2481944613, 2289028529, 846582864, 1024003631, 3457496236, 958218556, 2872479854, 1385329197, 2720560315, 2596604670, 4118717270, 3831528778, 4184700433, 130382780, 4221132295, 2492231677, 1349853675, 2674011478, 4230155413, 522369175, 1349933754, 2597169981, 1045439184, 3199517916, 3020468163, 618450142, 3282454786, 4061764145, 3477766529, 3055070885, 204747729, 834378094, 4046990333, 2141660100, 2118590568, 3537961224, 168082228, 652073352, 1284401985, 2826423363, 1325707607, 3731067674]
    [[723716237, 3333080471, 2787841228, 2113795590, 3230829213, 1843969928, 746019385, 2309594694, 2229839123, 489336389, 779150137, 1446103079, 889348879, 3745936412, 4281307413, 2173665908, 1717438567, 23839969, 2145022294, 1052302893, 1118678081, 1295195691, 3851408124, 3928788718, 2370966349, 561561790, 1051546482, 4224882136, 4206633849, 1329396735, 1801554849, 1138195103, 3123692201, 1122989246, 1270364612, 1593839649, 1463959047, 4229718824, 1709163888, 33163415, 2667737317, 2446567315, 4184687507, 3396502067, 483813678], [2220062804, 290402150, 576718487, 2834691630, 2746963147, 1838059276, 3070124527, 885776292, 1650043431, 3582686597, 220792127, 2535197382, 1699971710, 1898865050, 3798071317, 2827338235, 435208719, 1570192009, 2360615333, 515231859, 643500, 4187339056, 509908782, 486802097, 3677215785, 2921827920, 3237468089, 3427794385, 4032002655, 4093886240, 2687035724, 3077734642, 1170835917, 432938307, 2306967322, 2141534378, 2906718316, 1574298930, 4232243766, 240355277, 3647978042, 2451964704, 3514275296, 2490192022, 1350263172]]



```python
a = permutations[0][1]
b = permutations[1][1]

hash_val = (a * hash(tuple(["text", "word", "shingle", "advanced"]))+ b) % HASH_PRIME
print(hash_val)

hash_val = (a * hash(tuple(["text", "word", "shingle", "advanced"]))+ b) % HASH_PRIME
print(hash_val)
```

    1334347857
    1334347857



```python
hash_rdd = shingle_rdd.map(min_hasher)

for elem in hash_rdd.take(3):
    print(elem)
```

    (3, array([  61591645,  349817835,  569676983,    1599293,   29385299,
             46824109,   58918854,   13140136,  490863414,  172021836,
            642610371,  263500813,   71785523,  255385579,  715852370,
             53188389,  221433523,  116095602,  481157215,  202683349,
            231514234,  934947172,   45578852,  121412053,  339902106,
            116828610,  439822247,  327513644,   87449238,  389207925,
           1020938959,  273601212,  443052044,  137359580,  119793370,
            163442293, 1233259247,  249966399,   88224976,  460527268,
            300924346,  203156261,  942998696,  193567904,  338735817]))
    (6, array([ 5800535,  3519377,  2620441,  2404477, 17211334,  4159137,
            7747674,  2301261,  2521637, 33238753,  8657604,  4463566,
            3157822, 10269930, 17468121,  5525884,   624627,  1742699,
            2820437, 19874851,  1455725,  6627473,  2158930,   106103,
               7431, 11612108,  1597766, 22572504,  1641142,  4354797,
            8999150,   541015,  6508645,  1238272,   485276,  4401749,
           14358007, 11658496,  5241959,  4591721, 12825753,  2915677,
            2185725,  2520776,  7344211]))
    (7, array([ 78082176,    389471, 304166068, 104929478,  68490029,  69030234,
            51041631,   7718772, 172036243, 160556382, 215726405, 285420832,
            11759292,  18388850,  46765487,  26087129,  58258289,  27326873,
           141914304, 159652386,  78479427,  66113930,  23559821,  42272704,
            36501647,  44860424,   5175546,  33215735,  89558546, 228110406,
            85889090, 104423653,  74715564,  77159500,  22215641,   3151818,
            74239533, 126632636,  22313521,  88868238,  41779266,  71327844,
            45263437, 104357265,  29413811]))


# LSH


```python
BANDS: Final = 15
ROWS: Final = 3
THRESHOLD: Final = (1/BANDS) ** (1/ROWS)
print(f"Bands: {BANDS}, rows {ROWS}, threshold {THRESHOLD}")

def hash_func(row):
    sum = 0
    for e in row[1][0]:
        sum += e
    return (row[0], (int.from_bytes(hashlib.md5(str(sum).encode()).digest()[:4], byteorder="big"), row[1][1]))
```

    Bands: 15, rows 3, threshold 0.4054801330382267



```python
# returns (doc, band, hash)
hash_band_rdd = hash_rdd.flatMap(lambda x: [[(x[0], i % BANDS), hash] for i, hash in enumerate(x[1])]).groupByKey().cache()

for elem in hash_band_rdd.take(5):
    print(elem)
```

    ((42, 4), <pyspark.resultiterable.ResultIterable object at 0x7f9e9c80ff40>)
    ((103, 3), <pyspark.resultiterable.ResultIterable object at 0x7f9e9c80fbe0>)
    ((115, 11), <pyspark.resultiterable.ResultIterable object at 0x7f9ea6f1da90>)
    ((159, 1), <pyspark.resultiterable.ResultIterable object at 0x7f9e9c80ff10>)
    ((174, 0), <pyspark.resultiterable.ResultIterable object at 0x7f9e9c783310>)



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

    (11, 3247367347)
    233
    234
    (2, 3802146696)
    20787
    20789
    (8, 4101824626)
    32871
    29043
    (11, 3918574105)
    34135
    10717
    (14, 2024973377)
    84228
    84227
    (5, 1213592173)
    84285
    84278
    (1, 3768929074)
    12651
    75971
    (8, 1519802795)
    15053
    15054
    (14, 4037562956)
    218
    217
    (1, 1527845408)
    10954
    10953



```python
candidates = band_hashed.map(lambda x: (tuple(x[1]), 1)).reduceByKey(lambda a, b: ((float(a + b) / 10.0))).cache()

for elem in candidates.take(100):
    print(elem)
```

    ((233, 234), 0.11200000000000002)
    ((20787, 20789), 0.11120000000000001)
    ((32871, 29043), 1)
    ((34135, 10717), 0.11111120000000001)
    ((84228, 84227), 0.11111111119999999)
    ((84285, 84278), 1)
    ((12651, 75971), 1)
    ((15053, 15054), 0.11111111199999998)
    ((218, 217), 0.11111200000000002)
    ((10954, 10953), 0.11111111199999998)
    ((15853, 15852), 1)
    ((9831, 9832), 1)
    ((1209, 1210), 0.11111111199999998)
    ((1591, 1592), 0.11112)
    ((3410, 3409), 0.11111112000000001)
    ((9800, 9801), 0.11112)
    ((9847, 9846), 0.11120000000000001)
    ((9848, 9849), 0.11111112000000001)
    ((10965, 10966), 0.11200000000000002)
    ((15054, 31541, 15053), 1)
    ((18823, 18856), 1)
    ((26874, 26234), 0.11200000000000002)
    ((43308, 43322), 1)
    ((44896, 44897), 0.2)
    ((55474, 18737), 1)
    ((59977, 50805), 0.11112)
    ((68173, 20159), 1)
    ((71200, 56250), 1)
    ((77634, 67725), 0.2)
    ((10892, 10893), 0.11200000000000002)
    ((10896, 10897), 0.11111112000000001)
    ((12013, 12012), 0.11200000000000002)
    ((20789, 20787), 0.11120000000000001)
    ((41878, 42254), 0.11111111199999998)
    ((54257, 54258), 1)
    ((72388, 72389), 0.11112)
    ((216, 215), 0.1111111111112)
    ((30271, 10785), 0.11200000000000002)
    ((10622, 10621), 0.11111111199999998)
    ((11483, 11484), 0.11200000000000002)
    ((41230, 41249), 0.2)
    ((8869, 3213, 8878), 1)
    ((10360, 10361), 0.11200000000000002)
    ((12010, 12011), 0.11111200000000002)
    ((20122, 85548), 1)
    ((1590, 1589), 0.11200000000000002)
    ((8394, 8393), 0.11111200000000002)
    ((9833, 9834), 0.11111112000000001)
    ((10186, 10185), 1)
    ((10361, 10360), 0.11111111119999999)
    ((10457, 10456), 1)
    ((10893, 10892), 0.11111111119999999)
    ((10894, 3543), 0.11111120000000001)
    ((12014, 12015), 0.11112)
    ((34158, 9999), 1)
    ((48488, 48485), 0.11112)
    ((72389, 72388), 0.11111112000000001)
    ((87377, 87382), 0.12)
    ((1243, 1244), 0.11112)
    ((1560, 5197), 1)
    ((9837, 9836), 0.11111111199999998)
    ((9884, 9885), 0.11111120000000001)
    ((10621, 10622), 0.11120000000000001)
    ((12012, 12013), 0.11111111119999999)
    ((26037, 41279), 1)
    ((85886, 85914), 1)
    ((31695, 53158), 1)
    ((5340, 44474, 2170), 1)
    ((12516, 12515), 0.11112)
    ((64053, 64096), 0.11120000000000001)
    ((68366, 68359), 1)
    ((27645, 27646), 0.11112)
    ((31090, 32382), 0.11120000000000001)
    ((58712, 58711), 0.11120000000000001)
    ((1589, 1590), 0.11111111119999999)
    ((11093, 11094), 0.11200000000000002)
    ((13212, 13149), 0.12)
    ((30270, 10784), 0.111111111112)
    ((33475, 67066), 0.2)
    ((51348, 51349), 0.2)
    ((58893, 59763), 1)
    ((76164, 68234), 1)
    ((91844, 75489, 11927), 1)
    ((8869, 8878), 1)
    ((11796, 11795), 0.11120000000000001)
    ((14956, 42739, 33277), 1)
    ((22432, 26661), 1)
    ((29687, 13202), 0.2)
    ((91828, 91814), 1)
    ((26514, 26794), 0.2)
    ((11606, 11607), 1)
    ((58711, 58712), 0.11120000000000001)
    ((1205, 1206), 0.12)
    ((9841, 9840), 0.11112)
    ((9851, 9850), 1)
    ((11544, 11543), 0.12)
    ((11607, 11606), 0.11111112000000001)
    ((11820, 11819), 0.12)
    ((13103, 10772), 0.11200000000000002)
    ((26209, 26208), 1)



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
#candidate_sim = band_hashed.map(lambda x: [x[0], calc_jaccard(shingle_rdd.lookup(x[0][0]), shingle_rdd.lookup(x[0][1]))])

#for elem in candidate_sim.take(10):
#    print(elem)

for elem in cand:
    print(f"Candidates: {elem[0][0]}, {elem[0][1]}; similarity: ", end="")
    print(f"{calc_jaccard(shingle_dict.get(elem[0][0]), shingle_dict.get(elem[0][1]))}")
```

    Candidates: 233, 234; similarity: 0.8571428571428571
    Candidates: 20787, 20789; similarity: 0.7755102040816326
    Candidates: 32871, 29043; similarity: 0.0
    Candidates: 34135, 10717; similarity: 1.0
    Candidates: 84228, 84227; similarity: 1.0
    Candidates: 84285, 84278; similarity: 0.1242603550295858
    Candidates: 12651, 75971; similarity: 0.0
    Candidates: 15053, 15054; similarity: 0.9130434782608695
    Candidates: 218, 217; similarity: 1.0
    Candidates: 10954, 10953; similarity: 1.0
    Candidates: 15853, 15852; similarity: 0.34782608695652173
    Candidates: 9831, 9832; similarity: 0.5217391304347826
    Candidates: 1209, 1210; similarity: 1.0
    Candidates: 1591, 1592; similarity: 1.0
    Candidates: 3410, 3409; similarity: 1.0
    Candidates: 9800, 9801; similarity: 0.8709677419354839
    Candidates: 9847, 9846; similarity: 1.0
    Candidates: 9848, 9849; similarity: 1.0
    Candidates: 10965, 10966; similarity: 0.8
    Candidates: 15054, 31541; similarity: 0.0
    Candidates: 18823, 18856; similarity: 0.0
    Candidates: 26874, 26234; similarity: 1.0
    Candidates: 43308, 43322; similarity: 0.061452513966480445
    Candidates: 44896, 44897; similarity: 0.5714285714285714
    Candidates: 55474, 18737; similarity: 0.0
    Candidates: 59977, 50805; similarity: 1.0
    Candidates: 68173, 20159; similarity: 0.4930875576036866
    Candidates: 71200, 56250; similarity: 0.2692307692307692
    Candidates: 77634, 67725; similarity: 0.6533333333333333
    Candidates: 10892, 10893; similarity: 1.0
    Candidates: 10896, 10897; similarity: 1.0
    Candidates: 12013, 12012; similarity: 1.0
    Candidates: 20789, 20787; similarity: 0.7755102040816326
    Candidates: 41878, 42254; similarity: 1.0
    Candidates: 54257, 54258; similarity: 0.21212121212121213
    Candidates: 72388, 72389; similarity: 1.0
    Candidates: 216, 215; similarity: 1.0
    Candidates: 30271, 10785; similarity: 0.8571428571428571
    Candidates: 10622, 10621; similarity: 1.0
    Candidates: 11483, 11484; similarity: 1.0
    Candidates: 41230, 41249; similarity: 0.7104072398190046
    Candidates: 8869, 3213; similarity: 0.6666666666666666
    Candidates: 10360, 10361; similarity: 1.0
    Candidates: 12010, 12011; similarity: 1.0
    Candidates: 20122, 85548; similarity: 0.0
    Candidates: 1590, 1589; similarity: 1.0
    Candidates: 8394, 8393; similarity: 1.0
    Candidates: 9833, 9834; similarity: 1.0
    Candidates: 10186, 10185; similarity: 0.4782608695652174
    Candidates: 10361, 10360; similarity: 1.0
    Candidates: 10457, 10456; similarity: 0.40476190476190477
    Candidates: 10893, 10892; similarity: 1.0
    Candidates: 10894, 3543; similarity: 1.0
    Candidates: 12014, 12015; similarity: 1.0
    Candidates: 34158, 9999; similarity: 0.0
    Candidates: 48488, 48485; similarity: 1.0
    Candidates: 72389, 72388; similarity: 1.0
    Candidates: 87377, 87382; similarity: 0.8686868686868687
    Candidates: 1243, 1244; similarity: 1.0
    Candidates: 1560, 5197; similarity: 0.6744186046511628
    Candidates: 9837, 9836; similarity: 1.0
    Candidates: 9884, 9885; similarity: 1.0
    Candidates: 10621, 10622; similarity: 1.0
    Candidates: 12012, 12013; similarity: 1.0
    Candidates: 26037, 41279; similarity: 0.23255813953488372
    Candidates: 85886, 85914; similarity: 0.3041825095057034
    Candidates: 31695, 53158; similarity: 0.1407766990291262
    Candidates: 5340, 44474; similarity: 0.45454545454545453
    Candidates: 12516, 12515; similarity: 1.0
    Candidates: 64053, 64096; similarity: 0.8868894601542416
    Candidates: 68366, 68359; similarity: 0.22058823529411764
    Candidates: 27645, 27646; similarity: 1.0
    Candidates: 31090, 32382; similarity: 0.927710843373494
    Candidates: 58712, 58711; similarity: 0.8367346938775511
    Candidates: 1589, 1590; similarity: 1.0
    Candidates: 11093, 11094; similarity: 1.0
    Candidates: 13212, 13149; similarity: 1.0
    Candidates: 30270, 10784; similarity: 0.967741935483871
    Candidates: 33475, 67066; similarity: 0.5465116279069767
    Candidates: 51348, 51349; similarity: 0.42857142857142855
    Candidates: 58893, 59763; similarity: 0.07268170426065163
    Candidates: 76164, 68234; similarity: 0.0
    Candidates: 91844, 75489; similarity: 0.729381443298969
    Candidates: 8869, 8878; similarity: 0.5769230769230769
    Candidates: 11796, 11795; similarity: 0.7391304347826086
    Candidates: 14956, 42739; similarity: 0.45977011494252873
    Candidates: 22432, 26661; similarity: 0.23100303951367782
    Candidates: 29687, 13202; similarity: 0.5321100917431193
    Candidates: 91828, 91814; similarity: 0.75
    Candidates: 26514, 26794; similarity: 0.5135135135135135
    Candidates: 11606, 11607; similarity: 0.8571428571428571
    Candidates: 58711, 58712; similarity: 0.8367346938775511
    Candidates: 1205, 1206; similarity: 0.375
    Candidates: 9841, 9840; similarity: 1.0
    Candidates: 9851, 9850; similarity: 0.4444444444444444
    Candidates: 11544, 11543; similarity: 0.6
    Candidates: 11607, 11606; similarity: 0.8571428571428571
    Candidates: 11820, 11819; similarity: 0.6666666666666666
    Candidates: 13103, 10772; similarity: 0.7682926829268293
    Candidates: 26209, 26208; similarity: 0.16666666666666666
    Candidates: 84230, 84229; similarity: 1.0
    Candidates: 87189, 87195; similarity: 0.5166666666666667
    Candidates: 1210, 1209; similarity: 1.0
    Candidates: 3695, 781; similarity: 0.0
    Candidates: 10471, 43492; similarity: 0.0
    Candidates: 10970, 10969; similarity: 1.0
    Candidates: 11148, 11147; similarity: 1.0
    Candidates: 12008, 12009; similarity: 1.0
    Candidates: 19085, 16330; similarity: 0.0
    Candidates: 22374, 22386; similarity: 0.6
    Candidates: 26234, 26874; similarity: 1.0
    Candidates: 30289, 55602; similarity: 0.0
    Candidates: 40597, 40607; similarity: 0.25757575757575757
    Candidates: 47386, 46495; similarity: 0.0
    Candidates: 77473, 41814; similarity: 0.0
    Candidates: 86915, 44320; similarity: 0.0
    Candidates: 87648, 27666; similarity: 0.0
    Candidates: 9885, 9884; similarity: 1.0
    Candidates: 10943, 10942; similarity: 1.0
    Candidates: 84420, 84499; similarity: 0.5625
    Candidates: 12108, 12131; similarity: 0.5904761904761905
    Candidates: 27912, 41123; similarity: 0.0
    Candidates: 3409, 3410; similarity: 1.0
    Candidates: 47579, 47531; similarity: 0.7613636363636364
    Candidates: 66716, 81267; similarity: 0.0
    Candidates: 52175, 72938; similarity: 0.0
    Candidates: 217, 218; similarity: 1.0
    Candidates: 1207, 1208; similarity: 0.8076923076923077
    Candidates: 2242, 2243; similarity: 0.6060606060606061
    Candidates: 10942, 10943; similarity: 1.0
    Candidates: 10969, 10970; similarity: 1.0
    Candidates: 13149, 13212; similarity: 1.0
    Candidates: 13286, 13287; similarity: 0.30434782608695654
    Candidates: 31186, 8860; similarity: 0.0
    Candidates: 64414, 89742; similarity: 0.0
    Candidates: 78497, 3948; similarity: 0.4093567251461988
    Candidates: 86981, 86972; similarity: 0.41304347826086957
    Candidates: 12149, 12148; similarity: 0.43902439024390244
    Candidates: 26674, 10366; similarity: 0.33579335793357934
    Candidates: 37300, 37318; similarity: 0.620253164556962
    Candidates: 59149, 59145; similarity: 0.09090909090909091
    Candidates: 15309, 15298; similarity: 0.6582278481012658
    Candidates: 1208, 1207; similarity: 0.8076923076923077
    Candidates: 1228, 1227; similarity: 0.5
    Candidates: 75749, 3503; similarity: 0.0
    Candidates: 77855, 75679; similarity: 0.0
    Candidates: 29031, 27657; similarity: 0.5974025974025974
    Candidates: 41249, 41230; similarity: 0.7104072398190046
    Candidates: 1591, 1592; similarity: 1.0
    Candidates: 2572, 1693; similarity: 0.5666666666666667
    Candidates: 3416, 3415; similarity: 0.7692307692307693
    Candidates: 10841, 10840; similarity: 1.0
    Candidates: 11700, 11701; similarity: 0.3448275862068966
    Candidates: 39579, 39585; similarity: 0.3548387096774194
    Candidates: 41395, 41504; similarity: 0.29545454545454547
    Candidates: 80321, 929; similarity: 0.0
    Candidates: 92193, 92194; similarity: 0.2413793103448276
    Candidates: 9840, 9841; similarity: 1.0
    Candidates: 12515, 12516; similarity: 1.0
    Candidates: 24321, 1549; similarity: 0.2222222222222222
    Candidates: 29017, 28999; similarity: 0.05555555555555555
    Candidates: 10966, 10965; similarity: 0.8
    Candidates: 15816, 50560; similarity: 0.0
    Candidates: 37286, 37280; similarity: 0.11940298507462686
    Candidates: 50377, 50375; similarity: 0.12598425196850394
    Candidates: 26498, 48543; similarity: 0.0
    Candidates: 3543, 10894; similarity: 1.0
    Candidates: 9844, 9845; similarity: 0.5238095238095238
    Candidates: 9846, 9847; similarity: 1.0
    Candidates: 11415, 11416; similarity: 0.35714285714285715
    Candidates: 30501, 30225; similarity: 0.3316062176165803
    Candidates: 41800, 89409; similarity: 0.0
    Candidates: 3990, 24665; similarity: 0.0
    Candidates: 9662, 9661; similarity: 0.027972027972027972
    Candidates: 12690, 12691; similarity: 0.1686746987951807
    Candidates: 1693, 2572; similarity: 0.5666666666666667
    Candidates: 22578, 54007; similarity: 0.0
    Candidates: 2910, 2911; similarity: 0.4375
    Candidates: 10693, 10694; similarity: 0.3142857142857143
    Candidates: 32382, 31090; similarity: 0.927710843373494
    Candidates: 33604, 40874; similarity: 0.42473118279569894
    Candidates: 77014, 1160; similarity: 0.32917705735660846
    Candidates: 91739, 75534; similarity: 0.0
    Candidates: 10717, 34135; similarity: 1.0
    Candidates: 17913, 35097; similarity: 0.0
    Candidates: 37278, 32512; similarity: 0.3459915611814346
    Candidates: 9761, 9760; similarity: 0.3611111111111111
    Candidates: 11094, 11093; similarity: 1.0
    Candidates: 44829, 26599; similarity: 0.0
    Candidates: 74601, 74606; similarity: 0.08979591836734693
    Candidates: 84227, 84228; similarity: 1.0
    Candidates: 20537, 51621; similarity: 0.0
    Candidates: 68800, 68839; similarity: 0.6530612244897959
    Candidates: 5155, 5154; similarity: 0.3333333333333333
    Candidates: 84420, 84499; similarity: 0.5625
    Candidates: 8870, 8872; similarity: 0.46153846153846156
    Candidates: 59295, 59291; similarity: 0.10030395136778116
    Candidates: 10718, 34136; similarity: 1.0
    Candidates: 11147, 11148; similarity: 1.0
    Candidates: 15298, 15309; similarity: 0.6582278481012658
    Candidates: 18512, 18511; similarity: 0.25
    Candidates: 21200, 44221; similarity: 0.02456140350877193
    Candidates: 27657, 29031; similarity: 0.5974025974025974
    Candidates: 42254, 41878; similarity: 1.0
    Candidates: 68328, 77476; similarity: 0.0
    Candidates: 84045, 83911; similarity: 0.12885154061624648
    Candidates: 8393, 8394; similarity: 1.0
    Candidates: 18140, 9259; similarity: 0.0
    Candidates: 46791, 44772; similarity: 0.36363636363636365
    Candidates: 40036, 42051; similarity: 0.6732594936708861
    Candidates: 39284, 39283; similarity: 0.4166666666666667
    Candidates: 39283, 39284; similarity: 0.4166666666666667
    Candidates: 72526, 17955; similarity: 0.0
    Candidates: 77657, 60582; similarity: 0.0
    Candidates: 234, 233; similarity: 0.8571428571428571
    Candidates: 8878, 8869; similarity: 0.5769230769230769
    Candidates: 27646, 27645; similarity: 1.0
    Candidates: 64305, 46706; similarity: 0.0
    Candidates: 88738, 23994; similarity: 0.3877551020408163
    Candidates: 10668, 10669; similarity: 0.32
    Candidates: 11717, 11718; similarity: 0.5
    Candidates: 15852, 15853; similarity: 0.34782608695652173
    Candidates: 26189, 30384; similarity: 0.0
    Candidates: 31240, 58167; similarity: 0.0
    Candidates: 59812, 34969; similarity: 0.4716981132075472
    Candidates: 68150, 68141; similarity: 0.34579439252336447
    Candidates: 9849, 9848; similarity: 1.0
    Candidates: 12826, 12827; similarity: 0.2727272727272727
    Candidates: 48396, 48398; similarity: 0.2318181818181818
    Candidates: 9547, 9549; similarity: 0.3829787234042553
    Candidates: 22386, 22374; similarity: 0.6
    Candidates: 5250, 5238; similarity: 0.3409090909090909
    Candidates: 10770, 10771; similarity: 0.5416666666666666
    Candidates: 10785, 30271; similarity: 0.8571428571428571
    Candidates: 58523, 535; similarity: 0.0
    Candidates: 84499, 84420; similarity: 0.5625
    Candidates: 215, 216; similarity: 1.0
    Candidates: 5340, 2170; similarity: 0.48
    Candidates: 9663, 9658; similarity: 0.08229426433915212
    Candidates: 17693, 50398; similarity: 0.15207373271889402
    Candidates: 62767, 53627; similarity: 0.2971698113207547
    Candidates: 11484, 11483; similarity: 1.0
    Candidates: 35167, 35159; similarity: 0.28
    Candidates: 70297, 31387; similarity: 0.0
    Candidates: 87056, 87058; similarity: 0.193717277486911
    Candidates: 76975, 63968; similarity: 0.0
    Candidates: 29687, 47152; similarity: 0.2724252491694352
    Candidates: 30932, 30638; similarity: 0.0
    Candidates: 50805, 59977; similarity: 1.0
    Candidates: 81178, 84287; similarity: 0.0
    Candidates: 84079, 84013; similarity: 0.14285714285714285
    Candidates: 1592, 24991; similarity: 0.4
    Candidates: 5197, 1560; similarity: 0.6744186046511628
    Candidates: 35502, 58937; similarity: 0.0
    Candidates: 40077, 18150; similarity: 0.3764705882352941
    Candidates: 41888, 50265; similarity: 0.0
    Candidates: 1227, 1228; similarity: 0.5
    Candidates: 10620, 10619; similarity: 1.0
    Candidates: 86810, 31544; similarity: 0.0
    Candidates: 1282, 6; similarity: 0.6976744186046512
    Candidates: 26986, 1971; similarity: 0.0
    Candidates: 10082, 10083; similarity: 0.225
    Candidates: 11819, 11820; similarity: 0.6666666666666666
    Candidates: 13202, 29687; similarity: 0.5321100917431193
    Candidates: 18713, 37933; similarity: 0.3010752688172043
    Candidates: 32601, 32510; similarity: 0.7411764705882353
    Candidates: 74586, 39366; similarity: 0.0
    Candidates: 30845, 30757; similarity: 0.23090586145648312
    Candidates: 33202, 81446; similarity: 0.0
    Candidates: 33282, 8306; similarity: 0.0
    Candidates: 72464, 48137; similarity: 0.0
    Candidates: 76157, 87105; similarity: 0.0
    Candidates: 10493, 13006; similarity: 0.22935779816513763
    Candidates: 91844, 75489; similarity: 0.729381443298969
    Candidates: 60217, 59924; similarity: 0.01694915254237288
    Candidates: 1244, 1243; similarity: 1.0
    Candidates: 9830, 9829; similarity: 0.5476190476190477
    Candidates: 12009, 12008; similarity: 1.0
    Candidates: 25828, 61087; similarity: 0.3048780487804878
    Candidates: 30961, 51735; similarity: 0.0
    Candidates: 35378, 33628; similarity: 0.6333333333333333
    Candidates: 39582, 19860; similarity: 0.0
    Candidates: 47538, 47555; similarity: 0.22131147540983606
    Candidates: 20636, 20667; similarity: 0.5543478260869565
    Candidates: 62767, 62648; similarity: 0.3054545454545455
    Candidates: 63979, 40739; similarity: 0.0
    Candidates: 76493, 76494; similarity: 0.3333333333333333
    Candidates: 10953, 10954; similarity: 1.0
    Candidates: 70988, 40457; similarity: 0.0
    Candidates: 19099, 40937; similarity: 0.0
    Candidates: 32306, 31669; similarity: 0.5636363636363636
    Candidates: 33366, 33368; similarity: 0.543859649122807
    Candidates: 10772, 13103; similarity: 0.7682926829268293
    Candidates: 33683, 33696; similarity: 0.5348837209302325
    Candidates: 17679, 17664; similarity: 0.15254237288135594
    Candidates: 68873, 68875; similarity: 0.1927710843373494
    Candidates: 5737, 20866; similarity: 0.0
    Candidates: 9459, 9460; similarity: 0.4444444444444444
    Candidates: 12015, 12014; similarity: 1.0
    Candidates: 26694, 33926; similarity: 0.0
    Candidates: 40247, 40246; similarity: 0.1836734693877551
    Candidates: 47869, 43093; similarity: 0.5714285714285714
    Candidates: 57884, 57799; similarity: 0.1762820512820513
    Candidates: 68252, 15887; similarity: 0.0
    Candidates: 91416, 91434; similarity: 0.17204301075268819
    Candidates: 10619, 10620; similarity: 1.0
    Candidates: 27671, 26864; similarity: 0.0
    Candidates: 9460, 9459; similarity: 0.4444444444444444
    Candidates: 9778, 9779; similarity: 0.3625
    Candidates: 22641, 22622; similarity: 1.0
    Candidates: 41431, 31844; similarity: 0.3116883116883117
    Candidates: 47531, 47579; similarity: 0.7613636363636364
    Candidates: 56050, 56051; similarity: 0.06622516556291391
    Candidates: 9836, 9837; similarity: 1.0
    Candidates: 39640, 88886; similarity: 0.0
    Candidates: 91604, 13040; similarity: 0.0
    Candidates: 68477, 68481; similarity: 0.2534562211981567
    Candidates: 52135, 11193; similarity: 0.0
    Candidates: 24245, 75336; similarity: 0.0
    Candidates: 27256, 27308; similarity: 0.14814814814814814
    Candidates: 3415, 3416; similarity: 0.7692307692307693
    Candidates: 46474, 10451; similarity: 0.0
    Candidates: 214, 8881; similarity: 0.3076923076923077
    Candidates: 15054, 15053; similarity: 0.9130434782608695
    Candidates: 18187, 37480; similarity: 0.4406779661016949
    Candidates: 40867, 1355; similarity: 0.35714285714285715
    Candidates: 61428, 61432; similarity: 0.1643835616438356
    Candidates: 77795, 77794; similarity: 0.19047619047619047
    Candidates: 79560, 79533; similarity: 0.3522727272727273
    Candidates: 93799, 93779; similarity: 0.372972972972973
    Candidates: 77197, 5876; similarity: 0.24545454545454545
    Candidates: 27594, 15960; similarity: 0.17346938775510204
    Candidates: 43580, 76572; similarity: 0.0
    Candidates: 47024, 80441; similarity: 0.0
    Candidates: 27789, 27791; similarity: 0.30864197530864196
    Candidates: 58861, 61856; similarity: 0.06074766355140187
    Candidates: 88784, 75815; similarity: 0.0
    Candidates: 89466, 86396; similarity: 0.0
    Candidates: 9770, 9771; similarity: 0.40625
    Candidates: 11795, 11796; similarity: 0.7391304347826086
    Candidates: 26208, 26209; similarity: 0.16666666666666666
    Candidates: 24328, 86457; similarity: 0.0
    Candidates: 9801, 9800; similarity: 0.8709677419354839
    Candidates: 33043, 33042; similarity: 0.8823529411764706
    Candidates: 6034, 5935; similarity: 0.07462686567164178
    Candidates: 12011, 12010; similarity: 1.0
    Candidates: 35415, 81596; similarity: 0.0
    Candidates: 22622, 22641; similarity: 1.0
    Candidates: 30757, 30845; similarity: 0.23090586145648312
    Candidates: 77554, 77549; similarity: 0.21551724137931033
    Candidates: 57710, 86690; similarity: 0.0
    Candidates: 63962, 87943; similarity: 0.0
    Candidates: 68839, 68800; similarity: 0.6530612244897959
    Candidates: 10084, 10085; similarity: 0.42857142857142855
    Candidates: 34136, 10718; similarity: 1.0
    Candidates: 6077, 50527; similarity: 0.0
    Candidates: 9454, 9453; similarity: 0.21621621621621623
    Candidates: 48485, 48488; similarity: 1.0
    Candidates: 68294, 44290; similarity: 0.6363636363636364
    Candidates: 89428, 39570; similarity: 0.0
    Candidates: 248, 34892; similarity: 0.23220973782771537
    Candidates: 11518, 5152; similarity: 0.0
    Candidates: 91844, 11927; similarity: 0.5305486284289277
    Candidates: 93431, 93432; similarity: 0.2736842105263158
    Candidates: 61220, 60881; similarity: 0.0
    Candidates: 74690, 74710; similarity: 0.1671018276762402
    Candidates: 52189, 37581; similarity: 0.483695652173913
    Candidates: 9667, 45044; similarity: 0.0
    Candidates: 33368, 33366; similarity: 0.543859649122807
    Candidates: 34969, 59812; similarity: 0.4716981132075472
    Candidates: 10885, 10886; similarity: 0.3870967741935484
    Candidates: 16039, 81085; similarity: 0.5846153846153846
    Candidates: 5340, 44474; similarity: 0.45454545454545453
    Candidates: 19224, 53142; similarity: 0.0
    Candidates: 64096, 64053; similarity: 0.8868894601542416
    Candidates: 494, 86929; similarity: 0.0
    Candidates: 1592, 1591; similarity: 1.0
    Candidates: 81085, 16039; similarity: 0.5846153846153846
    Candidates: 92148, 92161; similarity: 0.03875968992248062
    Candidates: 12278, 12277; similarity: 0.5625
    Candidates: 24921, 24912; similarity: 0.27848101265822783
    Candidates: 36957, 36955; similarity: 0.30344827586206896
    Candidates: 37700, 5566; similarity: 0.0
    Candidates: 39461, 31053; similarity: 0.23809523809523808
    Candidates: 80509, 80508; similarity: 0.20774647887323944
    Candidates: 66119, 70132; similarity: 0.0
    Candidates: 66565, 61356; similarity: 0.0
    Candidates: 9775, 9774; similarity: 0.23214285714285715
    Candidates: 9834, 9833; similarity: 1.0
    Candidates: 80523, 12748; similarity: 0.0
    Candidates: 9845, 9844; similarity: 0.5238095238095238
    Candidates: 32084, 32058; similarity: 0.10548523206751055
    Candidates: 74700, 74702; similarity: 0.5187057633973711
    Candidates: 59829, 37482; similarity: 0.0
    Candidates: 67725, 77634; similarity: 0.6533333333333333
    Candidates: 40199, 40248; similarity: 0.024509803921568627
    Candidates: 11663, 11662; similarity: 0.24285714285714285
    Candidates: 60271, 16517; similarity: 0.0
    Candidates: 61940, 61829; similarity: 0.2565789473684211
    Candidates: 71058, 71060; similarity: 0.5454545454545454
    Candidates: 91814, 91828; similarity: 0.75
    Candidates: 43093, 47869; similarity: 0.5714285714285714
    Candidates: 15227, 32734; similarity: 0.0
    Candidates: 68844, 53930; similarity: 0.0
    Candidates: 1241, 1242; similarity: 0.25
    Candidates: 75489, 91844; similarity: 0.729381443298969
    Candidates: 9765, 9764; similarity: 0.3333333333333333
    Candidates: 22791, 37909; similarity: 0.35135135135135137
    Candidates: 22891, 70307; similarity: 0.0
    Candidates: 87361, 87353; similarity: 0.3554006968641115
    Candidates: 88884, 88830; similarity: 0.23469387755102042
    Candidates: 68382, 50686; similarity: 0.0
    Candidates: 10840, 10841; similarity: 1.0
    Candidates: 62218, 62252; similarity: 0.75
    Candidates: 3411, 3412; similarity: 0.3170731707317073
    Candidates: 9779, 9778; similarity: 0.3625
    Candidates: 68651, 68654; similarity: 0.16470588235294117
    Candidates: 72231, 71813; similarity: 0.0
    Candidates: 9764, 9765; similarity: 0.3333333333333333
    Candidates: 10897, 10896; similarity: 1.0
    Candidates: 11211, 19454; similarity: 0.0
    Candidates: 31669, 32306; similarity: 0.5636363636363636
    Candidates: 35505, 32938; similarity: 0.10891089108910891
    Candidates: 9850, 9851; similarity: 0.4444444444444444
    Candidates: 33696, 33683; similarity: 0.5348837209302325
    Candidates: 42051, 40036; similarity: 0.6732594936708861
    Candidates: 86321, 5566; similarity: 0.07954545454545454
    Candidates: 72514, 72521; similarity: 0.48905109489051096
    Candidates: 68434, 16461; similarity: 0.0
    Candidates: 11053, 63596; similarity: 0.1927710843373494
    Candidates: 25991, 59665; similarity: 0.0
    Candidates: 39537, 39307; similarity: 0.1595744680851064
    Candidates: 9778, 33735; similarity: 0.0
    Candidates: 84299, 84298; similarity: 0.7173333333333334
    Candidates: 86311, 85670; similarity: 0.3103448275862069
    Candidates: 10677, 10676; similarity: 0.27586206896551724
    Candidates: 40378, 40374; similarity: 0.3006993006993007
    Candidates: 11927, 75489; similarity: 0.6302895322939867
    Candidates: 59289, 3907; similarity: 0.16822429906542055
    Candidates: 2485, 67906; similarity: 0.008928571428571428
    Candidates: 91707, 91673; similarity: 0.13399503722084366
    Candidates: 5238, 5250; similarity: 0.3409090909090909
    Candidates: 9234, 93415; similarity: 0.0
    Candidates: 15055, 12926; similarity: 0.13986013986013987
    Candidates: 33243, 259; similarity: 0.0
    Candidates: 9829, 9830; similarity: 0.5476190476190477
    Candidates: 11718, 11717; similarity: 0.5
    Candidates: 47437, 60969; similarity: 0.0
    Candidates: 55307, 55767; similarity: 0.5797101449275363
    Candidates: 81089, 32608; similarity: 0.6941176470588235
    Candidates: 33042, 33043; similarity: 0.8823529411764706
    Candidates: 93619, 8666; similarity: 0.0
    Candidates: 8871, 8869; similarity: 0.5
    Candidates: 12277, 12278; similarity: 0.5625
    Candidates: 31561, 31683; similarity: 0.5507246376811594
    Candidates: 70765, 70763; similarity: 0.1862348178137652
    Candidates: 71398, 87433; similarity: 0.2222222222222222
    Candidates: 87030, 70323; similarity: 0.0
    Candidates: 8869, 3213; similarity: 0.6666666666666666
    Candidates: 34358, 8968; similarity: 0.0
    Candidates: 47073, 527; similarity: 0.0
    Candidates: 79949, 60095; similarity: 0.0
    Candidates: 11758, 11757; similarity: 0.23076923076923078
    Candidates: 13124, 13226; similarity: 0.0
    Candidates: 85761, 61992; similarity: 0.05120481927710843
    Candidates: 1766, 8381; similarity: 0.0
    Candidates: 86998, 86997; similarity: 0.2529411764705882
    Candidates: 62060, 27915; similarity: 0.17647058823529413
    Candidates: 1203, 1204; similarity: 0.22448979591836735
    Candidates: 62620, 63941; similarity: 0.0
    Candidates: 1225, 1226; similarity: 0.46153846153846156
    Candidates: 64688, 64684; similarity: 0.5413533834586466
    Candidates: 10694, 10693; similarity: 0.3142857142857143
    Candidates: 31622, 31623; similarity: 0.5
    Candidates: 30058, 83806; similarity: 0.4666666666666667
    Candidates: 71075, 71051; similarity: 0.33707865168539325
    Candidates: 1592, 63265; similarity: 0.12149532710280374
    Candidates: 20667, 20636; similarity: 0.5543478260869565
    Candidates: 24441, 24459; similarity: 0.6744186046511628
    Candidates: 58868, 77550; similarity: 0.0
    Candidates: 32608, 81089; similarity: 0.6941176470588235
    Candidates: 28986, 25442; similarity: 0.0
    Candidates: 40874, 33604; similarity: 0.42473118279569894
    Candidates: 76273, 77018; similarity: 0.0
    Candidates: 83384, 83382; similarity: 0.147239263803681
    Candidates: 31669, 62813; similarity: 0.296875
    Candidates: 2170, 44474; similarity: 0.6
    Candidates: 5131, 55401; similarity: 0.0
    Candidates: 10769, 10768; similarity: 0.29545454545454547
    Candidates: 11631, 11633; similarity: 0.21153846153846154
    Candidates: 85764, 48535; similarity: 0.0
    Candidates: 6140, 26996; similarity: 0.0
    Candidates: 31683, 31561; similarity: 0.5507246376811594
    Candidates: 6, 1282; similarity: 0.6976744186046512
    Candidates: 42307, 42316; similarity: 0.1270718232044199
    Candidates: 87382, 87377; similarity: 0.8686868686868687
    Candidates: 55802, 70623; similarity: 0.07142857142857142
    Candidates: 8184, 22524; similarity: 0.3148148148148148
    Candidates: 24488, 59067; similarity: 0.0
    Candidates: 86851, 31395; similarity: 0.0
    Candidates: 43295, 43134; similarity: 0.0
    Candidates: 80873, 67540; similarity: 0.5384615384615384
    Candidates: 29714, 29713; similarity: 0.23076923076923078
    Candidates: 88903, 15763; similarity: 0.0
    Candidates: 10771, 10770; similarity: 0.5416666666666666
    Candidates: 31574, 31700; similarity: 0.3333333333333333
    Candidates: 14956, 42739; similarity: 0.45977011494252873
    Candidates: 12131, 12108; similarity: 0.5904761904761905
    Candidates: 51767, 51768; similarity: 0.04513888888888889
    Candidates: 41203, 19218; similarity: 0.3181818181818182
    Candidates: 15085, 51358; similarity: 0.0
    Candidates: 63189, 63188; similarity: 0.3333333333333333
    Candidates: 84298, 84299; similarity: 0.7173333333333334
    Candidates: 14467, 29607; similarity: 0.24293785310734464
    Candidates: 70472, 70496; similarity: 0.41304347826086957
    Candidates: 57775, 58837; similarity: 0.0
    Candidates: 18034, 20641; similarity: 0.47761194029850745
    Candidates: 72299, 59728; similarity: 0.0
    Candidates: 66489, 85872; similarity: 0.0
    Candidates: 67452, 41362; similarity: 0.0



```python
#for elem in hash_bands_grouped_rdd.take(2):
#    print(elem)
    #for h in elem[1]:
    #    print(h)
```


```python
bands_rdd = hash_band_rdd.map(lambda x: [(x[0][1], x[1]), x[0][0]]) 

for elem in bands_rdd.take(5):
    print(elem)
```


```python
vector_bucket_rdd = bands_rdd.map(lambda x: [x[1], x[0][1]]).distinct()

for elem in vector_bucket_rdd.take(1):
    print(elem)
```


```python

```


```python

```


```python

```

# Exit Spark


```python
spark.stop()
```


```python

```
