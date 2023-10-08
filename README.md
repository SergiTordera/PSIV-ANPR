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

