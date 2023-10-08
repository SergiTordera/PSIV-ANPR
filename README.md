# PSIV-ANPR
Aquest projecte implementa un sistema de reconeixement automàtic de matrícules (ANPR) utilitzant la llibreria d'OpenCV i sistemes de MachineLearning SVM i OCR per el reconeixament de carcaters. L'objectiu principal és detectar i segmentar matricules espanyoles i poder fer el reconeixament de caracters.

## Codi
El projecte conté els següents arxius *.py* i *.sav*:
1. ``matriculas_v3.py``: Conte el codi principal del projecte, al executarlo es posa en funcionament tot el sistema de reconeixement automàtic de matrícules
2. ``models.py``: Conte les funcions necessaries per crear els models de SVM de lletres i digits.
3. ``DatasetMatriculaEspanyola.py``: Segementa la fotografia que conte els caracters amb la font de la matricula espanyola, i les guarda en una carpeta per poder crear els models posteriorment.
4. ``lletresv4(7).sav``: Model SVM per les lletres

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


