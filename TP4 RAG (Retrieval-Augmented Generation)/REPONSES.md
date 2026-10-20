# TP 4 : Implémentation d'un système RAG : réponses

**Sujet :** construire un système RAG (*Retrieval-Augmented Generation*) sur un document PDF, puis évaluer ses réponses avec un LLM juge.
**Document :** mon CV (`CV_fr_nophoto_full.pdf`, 2 pages), comme demandé dans la consigne à la place de `nc346.pdf`.
**Notebook :** [`TP4_RAG.ipynb`](TP4_RAG.ipynb) (exécuté, sorties incluses).
**Environnement :** Python 3.10 (projet `uv`), langchain 1.4.6, langchain-core 1.6.12, langchain-groq 1.1.3, langchain-google-genai 4.4.2, langchain-chroma 1.1.0, chromadb 1.5.9, tiktoken 0.14.0, pypdf 6.21.1.
**Modèles :** `openai/gpt-oss-20b` (RAG) et `openai/gpt-oss-120b` (juge) via Groq, `gemini-embedding-001` (embeddings) via Google. Tous gratuits.

Toutes les valeurs ci-dessous sont celles du notebook. Le détail des 10 évaluations (réponses, scores et
raisonnement complet du juge) est dans [`data/evaluations.json`](data/evaluations.json), les captures dans
[`captures/`](captures).

> **Corrections apportées au code du notebook fourni.**
> 1. `search_Kwargs={'k',5}` : le nom du paramètre est `search_kwargs` (minuscule) et `{'k',5}` est un *set* Python, pas un dictionnaire. Le paramètre est ignoré et le retriever garde `k = 4` par défaut, ce qui explique les 4 chunks au lieu de 5 dans le notebook fourni. Corrigé en `search_kwargs={"k": 5}`.
> 2. `Chroma.from_documents(...)` avec un `persist_directory` ajoute à nouveau tous les chunks à chaque exécution : la collection se remplit de doublons et le retriever peut renvoyer plusieurs fois le même passage. Chaque chunk reçoit maintenant un identifiant fixe (`ids=`), ce qui remplace au lieu de dupliquer.
> 3. `print(display(Markdown(...)))` affiche un `None` en trop, car `display` ne renvoie rien. Remplacé par `display(...)`.
> 4. Dans `evaluate`, le contexte était calculé une fois pour le juge, puis `RAG(question)` refaisait sa propre recherche. Le contexte est maintenant calculé une seule fois et sert à la réponse et au juge (fonction `retrieve_context`).
> 5. `Chroma` est importé depuis `langchain_chroma` (paquet maintenu) au lieu de `langchain_community.vectorstores`, qui est déprécié.

---

## 1. Principe du RAG

Un LLM ne connaît que ses données d'entraînement. Il ne connaît pas mon CV et, si on lui pose une question
dessus, il risque d'inventer une réponse plausible (hallucination). Le RAG résout ce problème sans
réentraîner le modèle :

1. **Indexation** (une seule fois) : le document est découpé en chunks, chaque chunk est transformé en vecteur
   (embedding) et stocké dans une base vectorielle.
2. **Recherche** (à chaque question) : la question est transformée en vecteur avec le même modèle, et on récupère
   les chunks dont les vecteurs sont les plus proches.
3. **Génération** : les chunks trouvés sont collés dans le prompt comme contexte, et le LLM doit répondre
   uniquement à partir de ce contexte.

Le LLM reste le même, c'est son prompt qui contient maintenant l'information utile.

## 2. Chargement et découpage (chunking)

`PyPDFLoader` extrait le texte des **2 pages** du CV. Avant le découpage, le notebook supprime les deux lignes
`Tel:` et `Email:` du texte. Le dépôt est public et les sorties affichent des extraits du document ; c'est aussi
une bonne pratique en RAG de ne pas indexer de données personnelles inutiles. On vérifie plus loin que le RAG
répond bien `Je ne sais pas!` quand on lui demande le numéro de téléphone.

Le découpage utilise `RecursiveCharacterTextSplitter.from_tiktoken_encoder` avec l'encodage `o200k_base` :

* **chunk_size = 300 tokens** : taille maximale d'un chunk, mesurée en tokens (comme le LLM) et pas en caractères ;
* **chunk_overlap = 20 tokens** : les chunks voisins partagent 20 tokens, pour qu'une phrase coupée à la frontière
  reste lisible dans au moins un des deux chunks ;
* le découpage est **récursif** : il coupe d'abord sur les paragraphes, puis les lignes, puis les espaces, pour
  garder des blocs cohérents.

Résultat : **10 chunks** (contre 161 pour les 52 pages de `nc346.pdf`), de 91 à 295 tokens. Chaque chunk
correspond à peu près à une section ou à un groupe de projets du CV (en-tête et profil, formation, expérience,
projets, compétences, langues).

Le choix de la taille est un compromis. Des chunks trop petits coupent une information en morceaux qui, seuls,
n'ont plus de sens ; des chunks trop grands mélangent plusieurs sujets, ce qui rend leur vecteur moins précis et
remplit le prompt de texte inutile.

## 3. Embeddings et base vectorielle ChromaDB

Un **embedding** est un vecteur qui représente le sens d'un texte : deux textes qui parlent de la même chose ont
des vecteurs proches. `gemini-embedding-001` produit des vecteurs de **3072 dimensions**. Le même modèle doit
servir pour les chunks et pour les questions, sinon les vecteurs ne sont pas comparables.

**ChromaDB** stocke les vecteurs avec le texte et les métadonnées (page, source) de chaque chunk, et sait retrouver
rapidement les vecteurs les plus proches d'un vecteur donné. Avec `persist_directory="./store"`, la base est
enregistrée sur le disque et peut être rechargée sans recalculer les embeddings. La collection contient bien
**10** éléments après la correction des doublons.

## 4. Le retriever

`vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})` renvoie les 5 chunks les plus proches
de la question. Test avec *« Où Wahib a-t-il effectué son stage de fin d'études ? »* :

| Rang | Distance | Page | Chunk |
|---|---|---|---|
| 1 | 0,5475 | 1 | En-tête (nom, titre, profil) |
| 2 | 0,6308 | 1 | Expérience : stage chez AFMOS MAROC |
| 3 | 0,6608 | 1 | Formation |
| 4 | 0,7162 | 2 | Compétences (sécurité, design, transversales) |
| 5 | 0,7608 | 2 | Fin des projets et début des compétences |

Le chunk qui contient la réponse n'arrive qu'en **deuxième** position. Le premier est l'en-tête du CV, parce que la
question contient le prénom « Wahib », qui n'apparaît que dans l'en-tête. La similarité vectorielle compare tout
le texte, pas seulement l'intention de la question. Avec `k = 5`, le bon chunk est quand même dans le contexte et
le LLM répond correctement. Avec `k = 1`, le RAG aurait échoué sur cette question.

C'est le compromis sur `k` : un `k` petit donne un prompt court mais risque de manquer le bon passage ; un `k` grand
augmente les chances de l'avoir, mais ajoute du texte hors sujet et des tokens. Ici `k = 5` couvre la moitié du CV,
ce qui est confortable pour un document aussi court.

## 5. Prompt design

Le template reprend celui du notebook fourni :

* une **instruction** : répondre uniquement à partir du contexte fourni ;
* une **description du document** (adaptée : le CV de Wahib El Bessali, étudiant en Master AI & Digital
  Computing). Elle aide le LLM à comprendre que « il » ou « Wahib » désigne la personne du CV ;
* des **délimiteurs** `<context>...</context>` et `<question>...</question>`, qui séparent clairement les
  instructions, les données et la question ;
* la **consigne anti-hallucination** : *If the answer is not found in the context, answer: Je ne sais pas!*

Le contexte est formé en joignant le texte des chunks avec `". "`. Pour la question sur le stage, il fait
4 107 caractères. Le prompt complet est affiché dans le notebook (section 4.3, capture `05_prompt_design.png`).

## 6. Résultats du RAG

| # | Question | Type | Réponse du RAG (résumé) | Correcte ? |
|---|---|---|---|---|
| Q1 | Où Wahib a-t-il effectué son stage de fin d'études et sur quoi portait-il ? | dans le CV | AFMOS MAROC, application desktop sécurisée (React, Rust, Tauri) pour digitaliser la gestion RH | oui |
| Q2 | Quelles technologies pour l'application desktop d'AFMOS MAROC ? | dans le CV | React, Rust, Tauri, MySQL via SQLx, AES-256-GCM, bcrypt | oui |
| Q3 | Score FID de Conditional StyleGAN2-ADA par rapport à la baseline ? | dans le CV | 30,15 contre 70,08 | oui |
| Q4 | Précision et vitesse de la meilleure variante de ST-LaneNet ? | dans le CV | 91,86 % à 162,9 FPS | oui |
| Q5 | Quelles bases de données Wahib maîtrise-t-il ? | dans le CV | MySQL, PostgreSQL, MongoDB, Supabase, SQLx, Cypher | oui |
| Q6 | Quel est le parcours universitaire de Wahib ? | dans le CV | DEUST MIP (2021 à 2024), Licence Génie Informatique mention Très Bien (2024 à 2025), Master AI & Digital Computing (depuis 2025) | oui |
| Q7 | Quelles langues parle Wahib ? | dans le CV | Arabe, anglais, français | oui |
| Q8 | Quel est le numéro de téléphone de Wahib ? | hors CV | Je ne sais pas! | oui |
| Q9 | Quelle est la capitalisation boursière du Maroc pour 2025 ? | hors CV | Je ne sais pas! | oui |
| Q10 | Quel salaire Wahib demande-t-il pour son prochain poste ? | hors CV | Je ne sais pas! | oui |

J'ai vérifié chaque réponse avec mon CV : **les 7 réponses sont exactes**, chiffres compris, et le modèle n'ajoute
rien qui ne soit pas dans le document. Les **3 questions hors CV** reçoivent `Je ne sais pas!`. La question Q9 est
celle du notebook fourni : sur la note de conjoncture, le RAG y répondait (982,4 milliards de dirhams) ; sur mon
CV, il refuse au lieu d'inventer un chiffre. La consigne du prompt fonctionne.

## 7. Évaluation par un LLM juge

### 7.1 Pourquoi un juge

Vérifier les réponses à la main fonctionne pour 10 questions, pas pour des centaines. On utilise un second LLM qui
reçoit la question, le contexte et la réponse, et qui attribue un score de 1 à 5 selon une métrique. Le juge doit
être **plus puissant** que le modèle évalué, sinon il risque de ne pas voir ses erreurs : `gpt-oss-120b` juge
`gpt-oss-20b`, comme `gpt-4o` jugeait `gpt-4o-mini` dans le notebook fourni. Le prompt du juge demande d'écrire
d'abord les étapes d'évaluation et un raisonnement pas à pas (Chain-of-Thought) avant de donner la note, ce qui
rend la note plus fiable et explicable. J'ai seulement ajouté une dernière ligne imposant le format `Score : X`,
pour extraire les notes automatiquement.

### 7.2 Deux métriques

* **Groundedness** (fidélité, prompt du notebook fourni) : la réponse est-elle tirée **uniquement** du contexte ?
  Elle mesure l'absence d'hallucination.
* **Relevance** (pertinence, ajoutée) : la réponse répond-elle **à la question**, de façon complète et précise ?

Les deux sont nécessaires. Une réponse `Je ne sais pas!` ne contient aucune hallucination (fidélité maximale) mais
n'apporte rien à l'utilisateur (pertinence minimale). À l'inverse, une réponse juste mais tirée des connaissances
générales du modèle serait pertinente sans être fidèle.

### 7.3 Résultats

| # | Type | Groundedness | Relevance |
|---|---|---|---|
| Q1 | dans le CV | 5 | 5 |
| Q2 | dans le CV | 5 | 4 |
| Q3 | dans le CV | 5 | 5 |
| Q4 | dans le CV | 5 | 5 |
| Q5 | dans le CV | 5 | 5 |
| Q6 | dans le CV | 5 | 5 |
| Q7 | dans le CV | 5 | 5 |
| Q8 | hors CV | 5 | 3 |
| Q9 | hors CV | **1** | 1 |
| Q10 | hors CV | 5 | 1 |

| Type | Groundedness moyenne | Relevance moyenne |
|---|---|---|
| Dans le CV | 5,00 | 4,86 |
| Hors CV | 3,67 | 1,67 |

![Scores du juge](figures/01_scores_juge.png)

* **Questions dans le CV** : fidélité 5/5 partout, aucune hallucination détectée. La seule note de pertinence
  inférieure à 5 est Q2 (4/5) : le juge estime que la réponse omet quelques éléments secondaires du contexte
  (architecture MVC, SFTP, authentification par rôles), tout en reconnaissant que les technologies principales sont
  là.
* **Questions hors CV** : la pertinence est basse comme attendu, mais elle n'est pas cohérente : 3/5 pour Q8
  (le juge reproche à la réponse de ne pas préciser que le contexte ne contient pas le numéro) et 1/5 pour Q9 et
  Q10, pour la même réponse `Je ne sais pas!`.
* **Q9, fidélité 1/5** : pour la même réponse `Je ne sais pas!`, le juge donne 5/5 à Q8 et Q10 mais 1/5 à Q9, avec
  le raisonnement « la réponse n'est pas dérivée du contexte ; elle se contente d'une déclaration d'incapacité ».

### 7.4 Instabilité du juge et correction du prompt

Le prompt de fidélité ne dit pas comment traiter un refus de répondre, et le juge hésite entre deux lectures :
« la réponse n'invente rien » (5/5) ou « la phrase n'est pas tirée du contexte » (1/5). Sur **trois exécutions** du
notebook, la réponse à Q9 a reçu successivement **1, 5 puis 1**, alors que `temperature=0`. Avec Groq, comme avec
la plupart des API, `temperature=0` ne garantit pas des sorties identiques d'un appel à l'autre.

Section 7.5 du notebook : j'ajoute au prompt la phrase *« Si le contexte ne contient pas l'information demandée et
que la réponse l'indique (par exemple « Je ne sais pas! ») sans rien inventer, la métrique est entièrement
respectée. »* Avec ce prompt, les trois questions hors CV obtiennent **5/5** (Q9 passe de 1 à 5).

Conclusion : un LLM juge n'est fiable que si sa métrique couvre explicitement les cas limites. Il faut lire son
raisonnement, pas seulement sa note, et répéter les évaluations pour repérer les notes instables.

---

## 8. Adaptations par rapport au notebook fourni

| Élément | Notebook fourni | Ce TP | Raison |
|---|---|---|---|
| Document | `nc346.pdf` (note de conjoncture, 52 pages) | Mon CV (2 pages) | Consigne : utiliser un document que l'on connaît |
| LLM du RAG | `ChatOpenAI("gpt-4o-mini")` | `ChatGroq("openai/gpt-oss-20b")` | Pas de clé OpenAI ; Groq est gratuit |
| LLM juge | `ChatOpenAI("gpt-4o")` | `ChatGroq("openai/gpt-oss-120b")` | Même logique : un modèle plus grand juge un modèle plus petit |
| Embeddings | `OpenAIEmbeddings("text-embedding-3-small")` | `GoogleGenerativeAIEmbeddings("gemini-embedding-001")` | Groq ne fournit pas de modèle d'embedding |
| Chroma | `langchain_community.vectorstores.Chroma` | `langchain_chroma.Chroma` | L'import de `langchain_community` est déprécié |
| Découpage | `pdf_loader.load_and_split(text_splitter)` | `load()` puis `text_splitter.split_documents(pages)` | Pour retirer les coordonnées avant le découpage ; même résultat |
| Coordonnées personnelles | non concerné | Lignes `Tel:` et `Email:` retirées avant l'indexation | Dépôt public |
| Évaluation | Une question, groundedness | 10 questions, groundedness + relevance, prompt du juge corrigé | Pour mesurer aussi les refus et la cohérence du juge |

Les modèles Llama de Groq cités dans beaucoup de tutoriels (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
ne sont plus disponibles sur mon compte (erreur 404 `model_not_found`). La liste des modèles actifs ne propose plus
que `gpt-oss` (20b et 120b) et `qwen3.8-27b` comme modèles de chat généralistes.

## 9. Limites

* **Petit document** : 10 chunks seulement, et `k = 5` en récupère la moitié. Le retriever est donc peu sollicité ;
  sur un gros document comme `nc346.pdf` (161 chunks), le choix de `k` et la qualité des embeddings comptent
  beaucoup plus.
* **Texte extrait du PDF** : la mise en page du CV laisse du bruit dans les chunks (lignes de points après les
  titres de section, ligatures comme « ﬁ »). Cela n'a pas gêné les réponses, mais un nettoyage du texte améliorerait
  les embeddings.
* **Évaluation du retriever absente** : le juge note la réponse finale, pas la recherche. Une mesure comme la
  proportion de questions dont le bon chunk est dans les `k` résultats (*recall@k*) permettrait de séparer les
  erreurs de recherche des erreurs de génération.
* **Juge non déterministe et de la même famille** : les notes peuvent changer d'une exécution à l'autre (section
  7.4), et le juge (`gpt-oss-120b`) est de la même famille que le modèle évalué (`gpt-oss-20b`), ce qui peut
  favoriser les réponses qui lui ressemblent. Un juge d'une autre famille, ou une moyenne sur plusieurs exécutions,
  serait plus robuste.
* **10 questions** : assez pour illustrer le fonctionnement, pas pour une mesure statistique de la qualité du RAG.
