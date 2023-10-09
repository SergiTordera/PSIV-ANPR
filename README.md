# PSIV-ANPR
Aquest projecte implementa un sistema de reconeixement automàtic de matrícules (ANPR) utilitzant la llibreria d'OpenCV i sistemes de MachineLearning SVM i OCR pel reconeixement de carreters. L'objectiu principal és detectar i segmentar matrícules espanyoles i poder fer el reconeixement de caràcters.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/99728a8d-9e2c-4718-b43a-07fee2f6707b)
## Codi
El projecte conté els següents arxius *.py* i *.sav*:
1. ``matriculas_v3.py``: Conté el codi principal del projecte, a l'executarlo es posa en funcionament tot el sistema de reconeixement automàtic de matrícules
2. ``models.py``: Conte les funcions necessàries per crear els models de SVM de lletres i dígits.
3. ``DatasetMatriculaEspanyola.py``: Segmenta la fotografia que conté els caràcters amb la font de la matrícula espanyola, i les guarda en una carpeta per poder crear els models posteriorment.
4. ``lletresv4(7).sav``: Model SVM per les lletres
## Detecció de Matricules
Per detectar la regió de la imatge on es troba la matrícula es poden utilitzar diferents tècniques com per exemple detecció de contorns probables o extracció de característiques a través de transformacions black-hat.

En un principi vam intentar reconèixer aquesta zona de la imatge a través de la detecció probable de contorns (rectangles) pertanyents a la matrícula, però aquesta tècnica era molt dependent de la perspectiva i entorn de la fotografia. Per tant, la vam descartar i vam decidir fer deteccions a través de l'extraccio de característiques amb la tècnica black-hat i aplicant dilates i erodes en la imatge resultant.

El propòsit de fer dilates i erodes era crear zones compactes separades en la imatge, d'aquesta forma aconseguíem totes les zones possibles on hi pogués haver-hi la matricula i només calia identificar quina d'aquestes era.

Per aconseguir la zona correcta es va implementar una funció ``find_region`` que buscava contorns rectangulars és a dir amb 4 costats i aspect ratio corresponent en la imatge dilatada. Aquesta implementació ha assolit trobar correctament totes les regions de les diferents imatges on hi havia la matrícula.

| Step 1 | Step 2 | ... | Step X|
| -------------| ------------- | -------------|------------- |
|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/bcf4c783-b62f-4cb4-9f6c-b8c16ce0bf81) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/1b424e79-c026-4b76-8189-d6a398316532)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/625a0e57-86bf-45b6-b471-7d6fffea7610)| ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/89f3ab0f-c652-4472-b6af-79d388eb5a61)|

## Segmentació de Caràcters

La segmentació de caràcters parteix de la imatge de la matrícula amb vora afegida per casos en que els nombres o lletres quedessin massa enganxats. A partir d’aquest punt es binaritza la matrícula i es realitza una operació ``findContours`` per trobar tots els possibles contorns on es troben els nombres i lletres.
A continuació, amb una sèrie de condicions es fa un primer filtratge per treure contorns que sabem que no seran caràcters de la matrícula. Es calcula el rectangle que conté cada contorn i apliquem el filtratge amb les condicions: un mínim i un màxim de la mida de l’àrea, amplada del rectangle menor a l’alçada i que els rectangles no es trobin massa a la vora de la imatge.

Una vegada es recullen aquests rectangles candidats, es treu també els rectangles que es troben dins d’un altre (com poden ser els casos de la lletra D, B, ... que tenen els contorns interiors i exterior). 

Finalment, s’agafen els 7 rectangles més grans dels que queden tenint en compte de treure el rectangle que pertany al distintiu europeu.

![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/8c74e3c3-38af-42c6-853b-50aedca7956e)


## Reconeixement de Caràcters 

Per abordar el reconeixement de caràcters en aquest projecte s'han utilitzat dues tècniques diferents.

### SVM
Per fer-ho cada caràcter es quantificarà mitjançant l'extracció de característiques. Per extreure aquestes característiques hem dividit la imatge de cada caràcter en regions 7x7 i sumem quants píxels pertanyen al caràcter en cada regió.

![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/ba2e2055-023f-4f98-a13d-22b7b41a27fc)


Un cop tenim el vector de característiques associat amb cada caràcter, passarem per un model d'aprenentatge automàtic per classificar i reconèixer cada element.


### OCR

## Anàlisis i Resultats

En aquest apartat s'explicaran els principals models que s'han provat, i els resultats obtinguts sobre aquests.

### Model SVM
Per el model SVM, Observem un accuracy molt bo en el model per la classificació de lletres (sobre el 98%) i un model no tan bo per els digits el qual només  aconsegueix un 80%.

![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/e458df8d-61c1-451a-85b8-90ca6a19e916)

Si observem les conseqüents matriu de confusió respecte els models de lletres i dígits, es veu molt clarament com les lletres tenen precisions molt elevades i són fàcilment classificables, en canvi, el model de dígits és més conflictius, ja que es confonen els dígits entre ells, principalment els problemes venen dels dígits 5 i 6, per tant, no es classifiquen correctament. Això es podria deure a la gran semblança entre els dígits mencionats els quals mantenen uns patrons de característiques semblants.

#### Matriu de confusió model lletres
![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/4d3ef771-9020-4dd8-b7fc-be6b3dc5da54)
#### Matriu de confusió model digits
![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/923e6722-1b7c-482b-baae-8e45c5d1b36e)

Per aquesta raó s'ha decidit que per al reconeixement i classificació dels dígits s'utilitzaria un model OCR, el qual incrementi la nostra accuracy, i es mantendrà la utilització del model SVM per al reconeixement i classificació de lletres.

### Model OCR


## Contributors
* Sergi Tordera - 1608676@uab.cat
* Eric Alcaraz - 1603504@uab.cat                
* Raul Dalgamoni - 1599225@uab.cat

PSIV-Grau de __Data Engineering__, 
UAB, 2023

