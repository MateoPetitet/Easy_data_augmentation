L'augmentation de données pour la classification s'effectue avec la bibliothèque [albumentations](https://albumentations.ai/), afin d'ajouter une diversité dans les images utilisées. Les transformations choisies l'ont été de sorte à être cohérentes avec le contexte sous-marin, le comportement des poissons et les caractéristiques optiques du milieu. Il y a ainsi 6 transformations successives pouvant être réalisées :

- rotation à 90° et/ou inversion de l'image
- découpe partielle et trasnlation
- rotation libre
- transformations optiques généralistes (aberration chromatique artificielle, mise à l'échelle, déformations...)
- changement de colorimétrie, luminosité...
- floutage
- ajout de bruit, particules, objets

Les transformations sont appliquées dans cet ordre, et une seule est choisie par niveau (incluant potentiellement son absence). Traduite en expression régulière, la pipeline de transformation est de la forme suivante :

`(Rotations90_flips|Ø)(Crop_move|Ø)(Rotation|Ø)(Transformations_diverses|Ø)(Couleur|Caractéristiques|Lumière|Ø)(Flous|Ø)(Bruits|Particules_Objets|Ø)`

L'objectif est de couvrir une grande variété de conditions d'éclairage, de turbidité et de météo, de position, comportement, emplacement et éloignement des poissons, de soucis optiques, de comportement des capteurs optiques et de donnée capturée. Ainsi, en combinant toutes ces transformations, nous pouvons créer 622 080 nouvelles images par image fournie, sans compter l'aléatoire existant dans presque chaque transformation. Nous allons détailler les transformations en utilisant l'image de référence suivante :
![[OG.jpg]]

## Rotations à 90° et inversions
Cette transformation vise à simuler toutes les positions possibles de l'individu par rapport à la caméra. Pour cette raison, des positions insensées éthologiquement n'ont pas été considérées (le poisson sur le dos notamment).

## Découpe et translation
Les découpes recadrent l'image aux trois-quart de ses dimensions d'origine. Cette découpe est soit centrée, soit extraite aléatoirement de l'image originale.
La translation, quant à elle, déplace l'image aléatoirement en la décentrant dans les deux axes d'une distance allant jusqu'à un quart des dimensions associées.

## Rotation
Il s'agit ici d'autoriser une rotation, dans le sens horaire ou anti-horaire, allant aléatoirement jusqu'à 45°.

## Transformations optiques
Beaucoup de transformations sont ici simulées :
- aberration chromatique, similaire à celle observable sur les bordures des images traitées (dû au grand angle des caméras utilisées)
- réduction d'échelle aléatoire, simulant un individu plus loin de la caméra allant de 10 à 90% de la taille originelle
- effet de gaufrage plus ou moins intense, accentuant certains contours
- simulation d'artefacts de compression de type `jpeg` ou `webp` d'intensité variable, pouvant occasionnellement apparaître dans la prise d'images
- augmentation aléatoire de la netteté, soulignant (comme pour le gaufrage) certains détails
- déformations, ajoutant de la variété aux images afin de refléter toute la diversité qui n'a pas été capturée par les échantillons choisis :
	- déformation élastique, ondulant de façon variable l'image
	- déformation en grille, réalisant jusqu'à 5 déformations longitudinales
	- déformation en grille élastique, associant les deux déformations précédentes
	- déformation dite « *Thin Plate Spline* »
- changements d'échelles divisant ou multipliant la dimension de l'image par deux
- dilatation ou érosion morphologique
- distorsions optiques similaires à celles d'un appareil photo classique ou grand angle
- perspective artificielle

## Variations optiques : couleur et luminosité
Plusieurs changements peuvent être effectués sur les propriétés optiques de l'image :
- changement aléatoire des valeurs de teinte, saturation et luminosité
- simulation d'éclairage d'intensité et angle variable
- gigue planckienne, simulant des changements de température de la luminosité
- modification aléatoire du contraste et de la luminosité
- gigue colorimétrique, modifiant aléatoirement la luminosité, le contraste, la saturation et la teinte
- glissement des valeurs RVB, déplaçant la valeur de chaque canal jusqu'à plus ou moins 30
- postérisation, réduisant aléatoirement le nombre de bits d'encodage des couleurs entre 3 et 5

## Floutage
Des flous variés permettant de simuler les déplacement des poissons ou les conditions de prise de vue sont utilisés :
- éloignement du foyer optique
- flou de zoom, similaire au rapprochement soudain d'un individu
- filtre médian
- flou de mouvement
- flou similaire à une vitre

## Bruit, particules, objets
Une collection de bruits et particules variant en forme, couleur et intensité est ajoutée afin de rendre le classifieur plus robuste aux obstacles et particules en suspensions :
- bruits :
	- gaussien
	- poivre et sel
	- de grenaille
- suppression de pixels aléatoires de certains canaux de couleur
- ajout d'une brume
- ajout de gravier
- ajout de pluie
- ajout d'ombres géométriques
- effet de neige
- ajout de halos lumineux
- ajout d'ombre plasmatique
- découpage de sections rectangulaires, allant d'une grande à plusieurs petites