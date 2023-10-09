# PSIV-ANPR
Aquest projecte implementa un sistema de reconeixement automàtic de matrícules (ANPR) utilitzant la llibreria d'OpenCV i sistemes de MachineLearning SVM i OCR per el reconeixament de carcaters. L'objectiu principal és detectar i segmentar matricules espanyoles i poder fer el reconeixament de caracters.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/99728a8d-9e2c-4718-b43a-07fee2f6707b)
## Codi
El projecte conté els següents arxius *.py* i *.sav*:
1. ``matriculas_v3.py``: Conte el codi principal del projecte, al executarlo es posa en funcionament tot el sistema de reconeixement automàtic de matrícules
2. ``models.py``: Conte les funcions necessaries per crear els models de SVM de lletres i digits.
3. ``DatasetMatriculaEspanyola.py``: Segementa la fotografia que conte els caracters amb la font de la matricula espanyola, i les guarda en una carpeta per poder crear els models posteriorment.
4. ``lletresv4(7).sav``: Model SVM per les lletres
## Detecció de Matricules
Per detectar la regio de la imatge on es troba la matricula es poden utilitzar diferents tecniques com per exemple deteccio de controns probables o extracció de carecteristiques a traves de transformacions black-hat.

En un principi vam intentar reconeixer aquesta zona de la imatge a traves de la deteccio probable de contonrs (rectangles) pertenyens a la matricula, pero aquesta tecnica era molt dependent de la prespectiva i entorn de la fotografia. Per tant la vam descartar i vam decidir fer deteccions a traves de la extraccio de carecteristiques amb la tecnica black-hat i aplicant dilates i erodes en la imatge resultant.

El proposit de fer dilates i erodes era crear zones compactes seperades en la imatge, d'aquesta forma aconseguiem totes les zones possibles on hi pugues haver-hi la matricula i nomes calia identificar quina d'aquestes era.

Per aconseguir la zona correcta es va implementar un funció ``find_region`` que buscava contorns  rectancuglars es a dirm amb 4 costats i aspect ratio corresponent en la imatge dilatada. Aquesta implementació ha aconseguit trobar correctament totes les regions de les diferentes imatges on hi havia la matricula.

| Step 1 | Step 2 | ... | Step X|
| -------------| ------------- | -------------|------------- |
|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/bcf4c783-b62f-4cb4-9f6c-b8c16ce0bf81) |![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/1b424e79-c026-4b76-8189-d6a398316532)|![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/625a0e57-86bf-45b6-b471-7d6fffea7610)| ![image](https://github.com/DCC-UAB/xnap-project-ed_group_16/assets/61145059/89f3ab0f-c652-4472-b6af-79d388eb5a61)|

## Segmentació de Caràcters

La segmentació de caràcters parteix de la imatge de la matricula amb vora afegida per casos en que els nombres o lletres quedessin massa enganxats. A partir d’aquest punt es binaritza la matricula i es realitza una operació ``findContours`` per trobar tots els possibles contorns on es troben els nombres i lletres.
A continuació, amb una sèrie de condicions es fa un primer filtratge per treure contorns que sabem que no seran caràcters de la matricula. Es calcula el rectangle que conté cada contorn y apliquem el filtratge amb les condicions: un mínim i un màxim de la mida de l’àrea, amplada del rectangle menor a l’alçada i que els rectangles no es trobin massa a la vora de la imatge.

Una vegada es recullen aquests rectangles candidats, es treu també els rectangles que es troben dins d’un altre (com poden ser els casos de la lletra D, B, ... que tenen els contorns interiors i exterior). 

Finalment, s’agafen els 7 rectangles més grans dels que queden tenint en comte de treure el rectangle que pertany al distintiu europeu.

![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/8c74e3c3-38af-42c6-853b-50aedca7956e)


## Reconeixement de Caràcters 

Per abordar el reconixemnet de caracters en aquest projecte s'han utilitzat dues tecniques diferents.

### SVM
Per fer-ho cada caràcter es quantificara mitjançant l'extracció de característiques . Per extreure aquestes carecteristiques hem dividit la imatge de cada caràcter en regions 7x7 i sumem quans pixels pertanyen al caracter en cada regió.

![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/ba2e2055-023f-4f98-a13d-22b7b41a27fc)


Un cop tenim el vector de característiques associat amb cada caràcter, passarem per un model d'aprenentatge automàtic per classificar i reconèixer cada element.


### OCR

## Anàlisis i Resultats

En aquest apartat s'explicaran els principals models que s'han provat, i els resultats obtinguts sobre aquests.

### Model SVM
Per el model SVM, Observem un accuracy molt bo en el model per la classificació de lletres (sobre el 98%) i un model no tant bo per els digits el qual nomes aconsegueix un 80%.

![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/e458df8d-61c1-451a-85b8-90ca6a19e916)


Si observem les consequents matriu de confusió respecte els models de lletres i digits, es veu molt clarament com les lletres tenen precisions molt elevades i són fàcilment classificables, en canvi,el model de digits es més conflictius, ja que es confonen els digits entre ells, principalment els problemes venen dels digits 5 i 6, per tant, no es classifiquen correctament. Això es podria deure a la gran semblança entre els digits mencionats els quals mantenen uns patrons de carecteristiques semblants.

#### Matriu de confusió model lletres
![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/4d3ef771-9020-4dd8-b7fc-be6b3dc5da54)
#### Matriu de confusió model digits
![image](https://github.com/SergiTordera/PSIV-ANPR/assets/61145059/923e6722-1b7c-482b-baae-8e45c5d1b36e)

Per aquesta raó s'ha decidit que per el reconeixement i classificació dels digits s'utilitzaria un model OCR, el qual incrementi la nostra accuracy, i es mantendira la utilització del model SVM per el reconeixment i classificaió de lletres.

### Model OCR


