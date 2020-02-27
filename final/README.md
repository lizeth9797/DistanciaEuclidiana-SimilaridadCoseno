

```python
import nltk
```


```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/karisauria/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import numpy as np    
from numpy import dot
from numpy.linalg import norm
from IPython.display import display
import csv
import pandas as pd
```


```python
stopwords.fileids()
```




    ['arabic',
     'azerbaijani',
     'danish',
     'dutch',
     'english',
     'finnish',
     'french',
     'german',
     'greek',
     'hungarian',
     'indonesian',
     'italian',
     'kazakh',
     'nepali',
     'norwegian',
     'portuguese',
     'romanian',
     'russian',
     'slovene',
     'spanish',
     'swedish',
     'tajik',
     'turkish']




```python
set(stopwords.words('spanish'))
```




    {'a',
     'al',
     'algo',
     'algunas',
     'algunos',
     'ante',
     'antes',
     'como',
     'con',
     'contra',
     'cual',
     'cuando',
     'de',
     'del',
     'desde',
     'donde',
     'durante',
     'e',
     'el',
     'ella',
     'ellas',
     'ellos',
     'en',
     'entre',
     'era',
     'erais',
     'eran',
     'eras',
     'eres',
     'es',
     'esa',
     'esas',
     'ese',
     'eso',
     'esos',
     'esta',
     'estaba',
     'estabais',
     'estaban',
     'estabas',
     'estad',
     'estada',
     'estadas',
     'estado',
     'estados',
     'estamos',
     'estando',
     'estar',
     'estaremos',
     'estará',
     'estarán',
     'estarás',
     'estaré',
     'estaréis',
     'estaría',
     'estaríais',
     'estaríamos',
     'estarían',
     'estarías',
     'estas',
     'este',
     'estemos',
     'esto',
     'estos',
     'estoy',
     'estuve',
     'estuviera',
     'estuvierais',
     'estuvieran',
     'estuvieras',
     'estuvieron',
     'estuviese',
     'estuvieseis',
     'estuviesen',
     'estuvieses',
     'estuvimos',
     'estuviste',
     'estuvisteis',
     'estuviéramos',
     'estuviésemos',
     'estuvo',
     'está',
     'estábamos',
     'estáis',
     'están',
     'estás',
     'esté',
     'estéis',
     'estén',
     'estés',
     'fue',
     'fuera',
     'fuerais',
     'fueran',
     'fueras',
     'fueron',
     'fuese',
     'fueseis',
     'fuesen',
     'fueses',
     'fui',
     'fuimos',
     'fuiste',
     'fuisteis',
     'fuéramos',
     'fuésemos',
     'ha',
     'habida',
     'habidas',
     'habido',
     'habidos',
     'habiendo',
     'habremos',
     'habrá',
     'habrán',
     'habrás',
     'habré',
     'habréis',
     'habría',
     'habríais',
     'habríamos',
     'habrían',
     'habrías',
     'habéis',
     'había',
     'habíais',
     'habíamos',
     'habían',
     'habías',
     'han',
     'has',
     'hasta',
     'hay',
     'haya',
     'hayamos',
     'hayan',
     'hayas',
     'hayáis',
     'he',
     'hemos',
     'hube',
     'hubiera',
     'hubierais',
     'hubieran',
     'hubieras',
     'hubieron',
     'hubiese',
     'hubieseis',
     'hubiesen',
     'hubieses',
     'hubimos',
     'hubiste',
     'hubisteis',
     'hubiéramos',
     'hubiésemos',
     'hubo',
     'la',
     'las',
     'le',
     'les',
     'lo',
     'los',
     'me',
     'mi',
     'mis',
     'mucho',
     'muchos',
     'muy',
     'más',
     'mí',
     'mía',
     'mías',
     'mío',
     'míos',
     'nada',
     'ni',
     'no',
     'nos',
     'nosotras',
     'nosotros',
     'nuestra',
     'nuestras',
     'nuestro',
     'nuestros',
     'o',
     'os',
     'otra',
     'otras',
     'otro',
     'otros',
     'para',
     'pero',
     'poco',
     'por',
     'porque',
     'que',
     'quien',
     'quienes',
     'qué',
     'se',
     'sea',
     'seamos',
     'sean',
     'seas',
     'sentid',
     'sentida',
     'sentidas',
     'sentido',
     'sentidos',
     'seremos',
     'será',
     'serán',
     'serás',
     'seré',
     'seréis',
     'sería',
     'seríais',
     'seríamos',
     'serían',
     'serías',
     'seáis',
     'siente',
     'sin',
     'sintiendo',
     'sobre',
     'sois',
     'somos',
     'son',
     'soy',
     'su',
     'sus',
     'suya',
     'suyas',
     'suyo',
     'suyos',
     'sí',
     'también',
     'tanto',
     'te',
     'tendremos',
     'tendrá',
     'tendrán',
     'tendrás',
     'tendré',
     'tendréis',
     'tendría',
     'tendríais',
     'tendríamos',
     'tendrían',
     'tendrías',
     'tened',
     'tenemos',
     'tenga',
     'tengamos',
     'tengan',
     'tengas',
     'tengo',
     'tengáis',
     'tenida',
     'tenidas',
     'tenido',
     'tenidos',
     'teniendo',
     'tenéis',
     'tenía',
     'teníais',
     'teníamos',
     'tenían',
     'tenías',
     'ti',
     'tiene',
     'tienen',
     'tienes',
     'todo',
     'todos',
     'tu',
     'tus',
     'tuve',
     'tuviera',
     'tuvierais',
     'tuvieran',
     'tuvieras',
     'tuvieron',
     'tuviese',
     'tuvieseis',
     'tuviesen',
     'tuvieses',
     'tuvimos',
     'tuviste',
     'tuvisteis',
     'tuviéramos',
     'tuviésemos',
     'tuvo',
     'tuya',
     'tuyas',
     'tuyo',
     'tuyos',
     'tú',
     'un',
     'una',
     'uno',
     'unos',
     'vosotras',
     'vosotros',
     'vuestra',
     'vuestras',
     'vuestro',
     'vuestros',
     'y',
     'ya',
     'yo',
     'él',
     'éramos'}




```python
tokenizer = RegexpTokenizer(r'\w+')
```


```python
with open('Tweets.csv', 'r') as file:
    reader = csv.reader(file)
    numero_tweets = 0
    for row in reader:
          print(f'\nNumero de Tweet: {numero_tweets+1}\n {row[0]}')
          numero_tweets += 1
    print(f'\n\nTotal de Tweets: {numero_tweets}')

text_01_tokens = tokenizer.tokenize(row[0].lower()) #tokenizar y quitar signos de puntuación
#print(text_01_tokens)

text_01_tokens_wout_stopwords = []

for word in text_01_tokens:
    if word not in stopwords.words('spanish'): text_01_tokens_wout_stopwords.append(word)

print(text_01_tokens_wout_stopwords)
```

    
    Numero de Tweet: 1
     Datos concretos sobre el Feminicidio en México, tomados del SESNP. El delito tiene una tendencia a la alza. La solución de López y su Fiscal Carnal: Desaparecerlo del Código Penal. Tenemos un gobierno que prefiere ponerse del lado del delincuente y allanarle el camino.
    
    Numero de Tweet: 2
     SOCIEDAD El Fiscal General explicó que su propuesta busca facilitar la  investigación por  feminicidio y proteger a las  víctimas.
    
    Numero de Tweet: 3
     Conferencia Presidente Se queda EU con más del 80% de comisiones de remesas: Profeco.
    
    Numero de Tweet: 4
     Conferencia Presidente Entrega Fiscalía 2 mil mdp a AMLO para premios de la rifa.
    
    Numero de Tweet: 5
     Durante la  ConferenciaPresidente Alejandro Gertz Manero, titular de la FGR Mexico, entregó 2 mil mdp al Instituto para Devolver al Pueblo lo Robado.
    
    Numero de Tweet: 6
     Las bonitas palabras de Emerson Fittipaldi sobre Fernando Alonso: Es un fenómeno muy talentoso que espero vuelva a la Fórmula 1 en 2021
    
    Numero de Tweet: 7
     De acuerdo con imágenes obtenidas por la Fiscalía capitalina, la mujer subió a Fátima a un vehículo blanco
    
    Numero de Tweet: 8
     El presidente López Obrador se comprometió a enfrentar con empresarios y sindicatos el problema de las pensiones de los trabajadores
    
    Numero de Tweet: 9
     Un estudiante en estado grave tras desalojo de padres de los 43 de Ayotzinapa y manifestantes en Chiapas
    
    Numero de Tweet: 10
     El fisco advierte por esquemas de operaciones realizadas entre 2017 y 2019 por más de 339,000 millones de pesos que involucran a 977 contribuyentes
    
    
    Total de Tweets: 10
    El fisco advierte por esquemas de operaciones realizadas entre 2017 y 2019 por más de 339,000 millones de pesos que involucran a 977 contribuyentes
    ['fisco', 'advierte', 'esquemas', 'operaciones', 'realizadas', '2017', '2019', '339', '000', 'millones', 'pesos', 'involucran', '977', 'contribuyentes']



```python
with open('Tweets.csv', 'r') as file:
    reader = csv.reader(file)
    numero_tweets = 0
    for row in reader:
          print(f'\nNumero de Tweet: {numero_tweets+1}\n {row[0]}')
          text_01_tokens = tokenizer.tokenize(row[0].lower()) 
          text_01_tokens_wout_stopwords = []
          for word in text_01_tokens:
            if word not in stopwords.words('spanish'): text_01_tokens_wout_stopwords.append(word)
          print(text_01_tokens_wout_stopwords)
          numero_tweets += 1
    print(f'\n\nTotal de Tweets: {numero_tweets}')


```

    
    Numero de Tweet: 1
     Datos concretos sobre el Feminicidio en México, tomados del SESNP. El delito tiene una tendencia a la alza. La solución de López y su Fiscal Carnal: Desaparecerlo del Código Penal. Tenemos un gobierno que prefiere ponerse del lado del delincuente y allanarle el camino.
    ['datos', 'concretos', 'feminicidio', 'méxico', 'tomados', 'sesnp', 'delito', 'tendencia', 'alza', 'solución', 'lópez', 'fiscal', 'carnal', 'desaparecerlo', 'código', 'penal', 'gobierno', 'prefiere', 'ponerse', 'lado', 'delincuente', 'allanarle', 'camino']
    
    Numero de Tweet: 2
     SOCIEDAD El Fiscal General explicó que su propuesta busca facilitar la  investigación por  feminicidio y proteger a las  víctimas.
    ['sociedad', 'fiscal', 'general', 'explicó', 'propuesta', 'busca', 'facilitar', 'investigación', 'feminicidio', 'proteger', 'víctimas']
    
    Numero de Tweet: 3
     Conferencia Presidente Se queda EU con más del 80% de comisiones de remesas: Profeco.
    ['conferencia', 'presidente', 'queda', 'eu', '80', 'comisiones', 'remesas', 'profeco']
    
    Numero de Tweet: 4
     Conferencia Presidente Entrega Fiscalía 2 mil mdp a AMLO para premios de la rifa.
    ['conferencia', 'presidente', 'entrega', 'fiscalía', '2', 'mil', 'mdp', 'amlo', 'premios', 'rifa']
    
    Numero de Tweet: 5
     Durante la  ConferenciaPresidente Alejandro Gertz Manero, titular de la FGR Mexico, entregó 2 mil mdp al Instituto para Devolver al Pueblo lo Robado.
    ['conferenciapresidente', 'alejandro', 'gertz', 'manero', 'titular', 'fgr', 'mexico', 'entregó', '2', 'mil', 'mdp', 'instituto', 'devolver', 'pueblo', 'robado']
    
    Numero de Tweet: 6
     Las bonitas palabras de Emerson Fittipaldi sobre Fernando Alonso: Es un fenómeno muy talentoso que espero vuelva a la Fórmula 1 en 2021
    ['bonitas', 'palabras', 'emerson', 'fittipaldi', 'fernando', 'alonso', 'fenómeno', 'talentoso', 'espero', 'vuelva', 'fórmula', '1', '2021']
    
    Numero de Tweet: 7
     De acuerdo con imágenes obtenidas por la Fiscalía capitalina, la mujer subió a Fátima a un vehículo blanco
    ['acuerdo', 'imágenes', 'obtenidas', 'fiscalía', 'capitalina', 'mujer', 'subió', 'fátima', 'vehículo', 'blanco']
    
    Numero de Tweet: 8
     El presidente López Obrador se comprometió a enfrentar con empresarios y sindicatos el problema de las pensiones de los trabajadores
    ['presidente', 'lópez', 'obrador', 'comprometió', 'enfrentar', 'empresarios', 'sindicatos', 'problema', 'pensiones', 'trabajadores']
    
    Numero de Tweet: 9
     Un estudiante en estado grave tras desalojo de padres de los 43 de Ayotzinapa y manifestantes en Chiapas
    ['estudiante', 'grave', 'tras', 'desalojo', 'padres', '43', 'ayotzinapa', 'manifestantes', 'chiapas']
    
    Numero de Tweet: 10
     El fisco advierte por esquemas de operaciones realizadas entre 2017 y 2019 por más de 339,000 millones de pesos que involucran a 977 contribuyentes
    ['fisco', 'advierte', 'esquemas', 'operaciones', 'realizadas', '2017', '2019', '339', '000', 'millones', 'pesos', 'involucran', '977', 'contribuyentes']
    
    
    Total de Tweets: 10



```python

```
