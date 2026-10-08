# TP1 Python (partie 1) — Réponses

## Exercice 1 — Système de notation pondérée

```python
identifiant = input("Identifiant de l'étudiant : ")

note_projet = float(input("Note du projet pratique (TP) : "))
note_ecrit = float(input("Note de l'examen écrit : "))

coeff_projet = int(input("Coefficient du projet : "))
coeff_ecrit = int(input("Coefficient de l'écrit : "))

moyenne = (note_projet * coeff_projet + note_ecrit * coeff_ecrit) / (coeff_projet + coeff_ecrit)

print("Pour l'étudiant " + identifiant + ", la moyenne finale du module est de " + str(moyenne) + " / 20.")
```

## Exercice 2 — Analyse d'une équation

```python
a = float(input("Coefficient a : "))
b = float(input("Coefficient b : "))
c = float(input("Coefficient c : "))

if a == 0:
    if b == 0:
        print("Équation dégénérée (a = 0 et b = 0).")
    else:
        print("Équation du premier degré (a = 0).")
    est_second_degre = False
else:
    est_second_degre = True

if est_second_degre:
    delta = b**2 - 4 * a * c
    print(f"Discriminant : Δ = {delta}")
    if delta > 0:
        print("Δ > 0 : l'équation admet 2 solutions réelles distinctes.")
    elif delta == 0:
        print("Δ = 0 : l'équation admet 1 solution réelle (double).")
    else:
        print("Δ < 0 : l'équation n'admet aucune solution réelle.")
```

## Exercice 3 — Contrôle de saisie et boucle d'affichage

```python
saisie = float(input("Veuillez saisir un nombre entier impair : "))

while saisie != int(saisie) or saisie % 2 == 0:
    if saisie != int(saisie):
        saisie = float(input("Erreur : Vous avez saisi un nombre décimal. Veuillez réessayer : "))
    else:
        saisie = float(input(f"Erreur : {int(saisie)} est un nombre pair. Veuillez réessayer : "))

nombre = int(saisie)

print(f"Saisie acceptée ! Voici les nombres de 0 à {nombre} :")
for i in range(nombre + 1):
    print(i)
```

## Exercice 4 — Traitement des notes

```python
def ajouter_bonus(note, bonus=2):
    note = note + bonus
    return note

x = 10
resultat = ajouter_bonus(x)
print(f"Valeur retournée : {resultat}")
print(f"x après l'appel : {x}")

notes = [12.0, 15.5]
nouvelles_notes = list(map(ajouter_bonus, notes))
print(f"Nouvelles notes : {nouvelles_notes}")

etudiants = ["Ali", "Sara"]
for etudiant, note in zip(etudiants, nouvelles_notes):
    print(f"{etudiant} : {note}")
```

## Exercice 5 — Saisie ordonnée et calcul de la médiane

```python
n = int(input("Nombre total de notes : "))

notes = []
for i in range(n):
    note = float(input(f"Note n°{i + 1} : "))
    while i > 0 and note < notes[-1]:
        note = float(input(f"Erreur : la note doit être >= {notes[-1]}. Note n°{i + 1} : "))
    notes.append(note)

print(f"Notes saisies : {notes}")

milieu = n // 2
if n % 2 == 0:
    print(f"Le nombre de notes ({n}) est pair.")
    mediane = (notes[milieu - 1] + notes[milieu]) / 2
else:
    print(f"Le nombre de notes ({n}) est impair.")
    mediane = notes[milieu]

print(f"La médiane est : {mediane}")
```

## Exercice 6 — Configuration d'un système multi-agents

```python
config_ia = {}

config_ia["agent_recherche"] = {
    "modele": "gpt-4",
    "temperature": 0.2,
    "outils": [("WebSearch", "v2.1"), ("Calculatrice", "v1.0")],
}

config_ia["agent_recherche"]["temperature"] = 0.5
print(config_ia)

print(config_ia["agent_recherche"]["outils"][1][0])
```
