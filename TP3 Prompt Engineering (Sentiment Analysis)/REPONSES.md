# TP 3 : Prompt Engineering pour l'analyse de sentiment : réponses

**Sujet :** classification d'avis de films IMDB en `positive` ou `negative` avec un LLM, sans entraînement.
**Notebook :** [`TP3_Sentiment_Analysis.ipynb`](TP3_Sentiment_Analysis.ipynb) (exécuté, sorties incluses).
**Environnement :** Python 3.12.0 (projet `uv`), pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.1, langchain-core 1.6.12, langchain-google-genai 4.4.2.
**Modèle :** `gemini-3.5-flash-lite` (API Gemini, offre gratuite).

Toutes les valeurs chiffrées ci-dessous sont celles produites par le notebook. Les figures se trouvent dans
[`figures/`](figures), les captures du notebook exécuté dans [`captures/`](captures) et les scores bruts dans
[`data/resultats.json`](data/resultats.json).

> **Corrections apportées au code de l'énoncé.**
> 1. `user_message_template` est utilisé partout mais n'est jamais défini : il a été défini comme ``"```{movie_review}```"``.
> 2. `import json` et l'import du modèle de chat manquent dans les imports.
> 3. Dans la boucle des évaluations répétées, le prompt CoT n'est jamais construit ni évalué :
>    `cot_few_shot_performance.append(few_shot_micro_f1)` recopie le score du few-shot. La boucle corrigée
>    construit les deux prompts avec les mêmes exemples et évalue les deux.
> 4. L'exemple de calcul du micro-F1 donne en réalité le F1 de la classe positive (voir 1.2).

---

## 1. Objectifs et métriques

### 1.1 La tâche

On attribue à chaque avis l'une des deux étiquettes `positive` ou `negative`. On n'entraîne aucun classifieur :
un LLM déjà entraîné reçoit des instructions en langage naturel (le prompt) et répond par l'étiquette. Le travail
consiste à trouver la formulation du prompt qui donne les meilleures prédictions, et à le **mesurer** sur des
exemples dont on connaît la bonne réponse.

### 1.2 Accuracy et micro-F1

* **Accuracy** : nombre de prédictions correctes divisé par le nombre total de prédictions.
* **Micro-F1** : on additionne les vrais positifs (TP), faux positifs (FP) et faux négatifs (FN) de **toutes les
  classes**, puis on calcule précision = TP / (TP + FP), rappel = TP / (TP + FN) et F1 = 2PR / (P + R).

**L'exemple de l'énoncé est faux.** Avec la matrice de confusion (lignes = prédit, colonnes = réel) :

| Prédit \ Réel | Positif | Négatif |
|---|---|---|
| **Positif** | 50 | 10 |
| **Négatif** | 5 | 100 |

l'énoncé prend TP = 50, FP = 10, FN = 5 et obtient 0,870. Ce sont les comptes de la **seule classe positive** :
0,870 est le F1 de la classe positive. Le micro-F1 additionne aussi la classe négative (TP = 100, FP = 5, FN = 10) :

| | Classe positive | Classe négative | Total (micro) |
|---|---|---|---|
| TP | 50 | 100 | **150** |
| FP | 10 | 5 | **15** |
| FN | 5 | 10 | **15** |

Micro-précision = micro-rappel = 150 / 165 = **0,909**, donc micro-F1 = **0,909**. Le notebook le vérifie avec
`scikit-learn` (Étape 1) :

| Métrique | Valeur |
|---|---|
| Accuracy | 0,909 |
| F1 de la classe positive (`average="binary"`) | 0,870 |
| F1 de la classe négative | 0,930 |
| Micro-F1 (`average="micro"`) | **0,909** |
| Macro-F1 (moyenne des F1 par classe) | 0,900 |

**Conséquence.** Quand chaque exemple reçoit exactement une étiquette, chaque erreur compte une fois comme FP
(pour la classe prédite) et une fois comme FN (pour la vraie classe). Le total des FP est donc égal au total des FN,
et le micro-F1 est **toujours égal à l'accuracy**. Dans ce TP, « micro-F1 » et « pourcentage de bonnes réponses »
sont la même chose. Cela reste un bon choix parce que le jeu de données est équilibré (voir 2.2) ; en cas de
déséquilibre, c'est le **macro-F1** qui donnerait une mesure plus nuancée, car il donne le même poids à chaque classe.

---

## 2. Constitution des données

### 2.1 Source

L'énoncé charge `load_dataset("imdb")` depuis Hugging Face. Le notebook utilise le fichier fourni
`dataset_imdb.zip`, qui contient la version Kaggle du même corpus (avec une version récente de `huggingface_hub`,
l'identifiant court `"imdb"` n'est d'ailleurs plus accepté ; il faudrait écrire `"stanfordnlp/imdb"`).

| | Énoncé (`load_dataset`) | Fichier fourni |
|---|---|---|
| Avis utilisés | 25 000 (partie `train`) | 50 000 (`train` et `test` réunis) |
| Colonnes | `text`, `label` (0 / 1) | `review`, `sentiment` (`positive` / `negative`) |
| Doublons | non vérifié | **418** lignes en double |

`pandas` lit le zip directement. On renomme `review` en `text` (pour que le code de l'énoncé fonctionne tel quel),
on supprime les doublons et on recrée `label`. Il reste **49 582 avis**.

Supprimer les doublons n'est pas un détail : sans cela, un même avis pourrait se retrouver à la fois dans les
exemples montrés au modèle et dans les exemples de référence qui servent à le noter.

### 2.2 Équilibre des classes

| Sentiment | Avis |
|---|---|
| positive | 24 884 |
| negative | 24 698 |

Le jeu est équilibré à 50,2 % / 49,8 % ([figure](figures/01_equilibre_classes.png)). C'est la condition donnée par
l'énoncé pour utiliser le micro-F1, et cela rend le score interprétable : un modèle qui répondrait toujours la même
étiquette n'obtiendrait qu'environ 0,5.

### 2.3 Exemples et exemples de référence (gold examples)

`train_test_split(df, test_size=0.2, random_state=42)` sépare les données en deux réservoirs **disjoints** :

| Réservoir | Taille | Rôle |
|---|---|---|
| `examples_df` | 39 665 | on y tire les exemples **montrés** au modèle dans les prompts few-shot |
| `gold_examples_df` | 9 917 | on y tire les avis qui servent **uniquement à évaluer** les prompts |

Les deux ne se recouvrent pas : le modèle n'est jamais évalué sur un avis qui lui a été donné en exemple.
`random_state=42` rend la séparation identique à chaque exécution.

On retient **20 exemples de référence** (`.sample(20, random_state=42)`), enregistrés en JSON
([`data/gold_examples.json`](data/gold_examples.json)) au format `[{"text": ..., "sentiment": ...}, ...]`, qui est
aussi le format lu par `evaluate_prompt`. La graine fixe garantit que tous les prompts passent le même « examen ».

Remarque : ce tirage contient **12 avis négatifs et 8 positifs**. Le réservoir est équilibré mais un échantillon de
20 ne l'est pas forcément.

---

## 3. Élaboration des prompts

### 3.1 Format des messages

Un modèle de chat reçoit une **liste de messages**, chacun avec un rôle imposé par l'API : `system` (les consignes),
`user` (ce que dit l'utilisateur) et `assistant` (ce que le modèle a répondu). Les noms des rôles sont fixes ; seul
leur contenu est libre. LangChain traduit ce format pour chaque fournisseur (Gemini appelle le rôle assistant
`model` et place le message système à part), ce qui permet de garder le code de l'énoncé, écrit pour OpenAI.

Chaque avis est envoyé entre triples backticks (`user_message_template`) pour que le modèle distingue clairement le
texte de l'avis des consignes.

### 3.2 Prompt 1 : zero-shot

Uniquement un message système : la tâche, le délimiteur, et l'obligation de répondre par un seul mot sans
explication. Cette contrainte de format rend la réponse facile à analyser automatiquement.

### 3.3 Prompt 2 : few-shot

Même message système, suivi d'exemples. `create_prompt` transforme chaque exemple en **deux messages** : l'avis dans
un message `user` et son étiquette dans un message `assistant`. On fabrique ainsi un faux historique de conversation
dans lequel le modèle « a déjà » répondu correctement, en un seul mot, à plusieurs avis. Le modèle n'a pas de mémoire
entre deux appels et ne distingue pas un historique réel d'un historique construit : la suite la plus probable est
de continuer de la même façon. Séparer l'avis et l'étiquette dans deux messages montre exactement le format et la
position de la réponse attendue.

Ce faux historique est renvoyé à **chaque appel**, ce qui rend les appels few-shot plus longs et plus coûteux en
tokens que les appels zero-shot.

**Masques booléens.** `positive_reviews = (examples_df.sentiment == "positive")` n'est pas la liste des avis
positifs mais un masque `True` / `False` d'une valeur par ligne. Sa forme est donc `(39665,)` pour les deux masques
(l'énoncé affiche `(20000,)` pour les deux) ; le nombre d'avis positifs s'obtient avec `.sum()` : 19 906 positifs et
19 759 négatifs.

**Biais des exemples.** Tirer des exemples au hasard ne suffit pas :

* **biais de l'étiquette majoritaire** : si les exemples sont surtout positifs, le modèle a tendance à répondre
  « positive » plus souvent ;
* **biais de récence** : les derniers exemples du prompt pèsent davantage dans la réponse.

`create_examples(dataset, n)` tire **n exemples de chaque classe** (contre le premier biais), les concatène puis les
mélange avec `.sample(2 * n, replace=False)` (contre le second : un tirage sans remise de toutes les lignes revient
à une permutation aléatoire). Le mélange ne garantit pas l'alternance : lors d'un premier essai, les 4 exemples
étaient sortis dans l'ordre négatif, négatif, positif, positif, cas typique où le biais de récence pourrait jouer.
C'est une raison de plus pour répéter l'évaluation sur plusieurs tirages.

Le notebook fixe `np.random.seed(42)` avant les tirages pour qu'ils soient reproductibles (`.sample` sans
`random_state` utilise le générateur global de NumPy).

### 3.4 Prompt 3 : Chain-of-Thought

Les exemples sont **les mêmes** que pour le few-shot ; seul le message système change, avec deux étapes de
raisonnement (lire attentivement l'avis, estimer la probabilité qu'il soit positif à partir du sentiment global) et
un rappel du format de réponse. Ne changer qu'un élément à la fois fait de la comparaison few-shot / CoT une
expérience contrôlée : une différence de score ne pourrait venir que des instructions ajoutées.

Limite de ce prompt : il demande de réfléchir étape par étape mais aussi de ne pas expliquer la réponse. Or le
Chain-of-Thought fonctionne parce que le modèle **écrit** son raisonnement avant de conclure. Ici, il n'a aucune
place pour le faire ; il s'agit plutôt d'un prompt avec des consignes de lecture plus détaillées que d'un vrai CoT.

---

## 4. Évaluation des prompts

### 4.1 Le modèle et ses réglages

| | Énoncé | Ce TP | Raison |
|---|---|---|---|
| Modèle | `gpt-5.2` (`ChatOpenAI`) | `gemini-3.5-flash-lite` (`ChatGoogleGenerativeAI`) | l'API OpenAI est payante ; Gemini a une offre gratuite. `gemini-2.5-flash-lite` a été refusé (404, plus disponible pour les nouveaux utilisateurs) |
| Raisonnement interne | sans objet | `thinking_level="minimal"` | voir ci-dessous |
| Température | `temperature=0` | non fixée | Gemini 3 impose ses paramètres d'échantillonnage et ignore `temperature` (LangChain l'indique par un avertissement) |
| Débit | non limité | `InMemoryRateLimiter(0.2 requête/s)`, `max_retries=6` | rester sous les limites de l'offre gratuite |

**`thinking_level="minimal"`.** Les modèles Gemini 3 raisonnent par défaut avant de répondre. Un simple appel de
test prenait alors plus d'une minute, la réponse était plus longue (« **Bonjour** (standard/formal) ... You can also
say **Salut** ... » au lieu de « Bonjour ! »), et `.content` était renvoyé sous forme de **liste de blocs** avec une
signature de raisonnement au lieu d'une chaîne. Au niveau minimal, un appel prend environ 1,2 s et la réponse est
directe. Ce réglage ne fausse pas la comparaison : le CoT de l'énoncé est uniquement une consigne dans le prompt, et
les trois prompts sont évalués dans les mêmes conditions.

**Modification de `evaluate_prompt`.** La réponse est lue avec `response.text` au lieu de `response.content`.
Avec Gemini 3, `.content` peut être une liste, et `response_content.strip().lower()` lèverait une exception à chaque
avis ; le `try / except` de l'énoncé ferait alors passer silencieusement tous les avis. `.text` renvoie toujours une
chaîne. Le reste de la fonction est celui de l'énoncé, notamment le test de `negative` **avant** `positive`.

### 4.2 Résultats

**Un seul tirage d'exemples** (4 exemples, 20 avis de référence, [figure](figures/02_scores_tirage_unique.png)) :

| Prompt | Micro-F1 |
|---|---|
| Zero-shot | 1,00 |
| Few-shot | 1,00 |
| CoT | 1,00 |

**Évaluations répétées** (8 exemples par tirage, 20 avis de référence, [figure](figures/03_distribution_scores_runs.png)) :

| Prompt | Tirages | Micro-F1 moyen | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Zero-shot | 1 | 1,000 | 0,000 | 1,00 | 1,00 |
| Few-shot | 4 | 1,000 | 0,000 | 1,00 | 1,00 |
| CoT | 4 | 1,000 | 0,000 | 1,00 | 1,00 |

Soit 11 évaluations de 20 avis (220 prédictions) sans aucune erreur. C'est le même résultat que l'énoncé avec
`gpt-5.2` (1,0 pour les trois prompts, écart-type 0,0).

**Nombre de tirages.** L'énoncé fait 10 tirages. L'offre gratuite de Gemini est limitée à **500 requêtes par jour**
pour ce modèle (erreur `429 RESOURCE_EXHAUSTED`, quota `GenerateRequestsPerDayPerProjectPerModel-FreeTier`), et
chaque tirage coûte 40 appels (2 prompts × 20 avis). Le quota a été atteint après 4 tirages, en comptant les essais
préalables. Les 8 évaluations obtenues étant toutes à 1,0, on s'est arrêté à 4 tirages. Pour ne pas perdre de
résultats en cas d'interruption, le notebook enregistre chaque score dans
[`data/evaluations_cache.json`](data/evaluations_cache.json) dès qu'il est calculé.

### 4.3 Interprétation

* **Les trois prompts sont équivalents sur ce jeu.** Les avis IMDB sont en général très explicites (« waste of
  time », « one of the best films ») et un LLM récent a vu énormément de textes de ce type : la tâche est facile pour
  lui, même sans exemple. Les exemples et les consignes de raisonnement n'ont rien à corriger.
* **La variabilité existe quand même.** Lors d'un premier essai exploratoire (autre tirage de 4 exemples, sans graine
  fixée), le prompt few-shot avait obtenu **0,95** (une erreur sur 20) alors que le zero-shot et le CoT avaient
  obtenu 1,0 avec les mêmes avis. Une seule erreur suffit à faire bouger le score de 0,05 : c'est exactement pourquoi
  l'énoncé insiste sur les évaluations répétées et sur la prise en compte de la variabilité due au tirage.
* **Choix du prompt.** À performance égale, le **zero-shot** est le meilleur choix ici : il est le plus court, le
  moins coûteux en tokens (le few-shot renvoie 8 avis complets à chaque appel) et il ne dépend d'aucun tirage.

---

## 5. Limites

1. **20 exemples de référence, c'est peu.** La résolution du score est de 0,05 et un 20 / 20 ne prouve pas que le
   modèle est parfait : l'intervalle de confiance exact à 95 % (Clopper-Pearson) d'une accuracy de 20 / 20 est
   **[0,83 ; 1]**. L'énoncé le reconnaît : ce nombre sert seulement à limiter le temps d'exécution. Avec un jeu de
   référence plus grand, des erreurs apparaîtraient (sarcasme, avis mitigés), et c'est sur ces cas difficiles que les
   prompts pourraient se départager. Avec 500 requêtes par jour, l'évaluation des trois prompts sur 100 avis
   coûterait déjà 300 appels.
2. **4 tirages au lieu de 10**, à cause du quota journalier (voir 4.2).
3. **Reproductibilité partielle.** La température étant ignorée par Gemini 3, deux exécutions identiques peuvent
   donner une réponse différente sur un avis limite. Les tirages d'exemples sont reproductibles (graine fixée), pas
   les réponses du modèle.
4. **Le prompt CoT n'en est pas vraiment un**, puisqu'il interdit au modèle d'écrire son raisonnement (voir 3.4).
5. **Comparaison avec l'énoncé.** Le modèle n'est pas le même (`gemini-3.5-flash-lite` au lieu de `gpt-5.2`) et les
   20 avis de référence non plus (autre version du corpus, doublons supprimés) ; seules les conclusions sont
   comparables, pas les avis un à un.

---

## 6. Conclusion

La démarche du TP est celle d'une expérience : fixer une métrique (micro-F1, ici égal à l'accuracy), séparer les
exemples montrés au modèle des exemples qui servent à le noter, ne changer qu'un élément du prompt à la fois, et
répéter les mesures pour tenir compte du hasard des tirages. Sur 20 avis IMDB, `gemini-3.5-flash-lite` obtient un
micro-F1 de 1,0 avec les trois prompts, comme `gpt-5.2` dans l'énoncé : la classification de sentiment d'avis de films
est une tâche que les LLM actuels réussissent sans exemple. Pour départager réellement des prompts, il faudrait un
jeu de référence beaucoup plus grand ou des avis plus ambigus.
