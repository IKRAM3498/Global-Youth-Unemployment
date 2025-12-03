# 📊 Projet : Analyse et Prédiction du Taux de Chômage des Jeunes (Global Youth Unemployment)

## 1. Introduction

Ce rapport présente les étapes initiales d'un projet d'analyse de données visant à modéliser et à prédire les tendances du chômage des jeunes à l'échelle mondiale. Le jeu de données utilisé couvre une période historique étendue et inclut des observations par pays et par année.

---

## 2. Sélection du Jeu de Données

| Caractéristique | Détail |
| :--- | :--- |
| **Nom du Fichier** | `youth_unemployment_global.csv` |
| **Source** | Données agrégées (simulées pour l'exercice) basées sur des indicateurs macroéconomiques mondiaux. |
| **Thématique** | Évolution temporelle et géographique du taux de chômage des jeunes (généralement défini comme la population active âgée de 15 à 24 ans). |
| **Pertinence** | Le chômage des jeunes est un indicateur socio-économique majeur. La complexité réside dans l'analyse des séries temporelles, des facteurs géopolitiques et des disparités régionales, ce qui en fait un jeu de données non trivial pour la modélisation prédictive. |

---

## 3. Définition de la Problématique

Le projet se concentre sur une problématique de **modélisation prédictive** basée sur des données historiques et géographiques.

### Objectif Principal

Développer un modèle capable de prédire le **taux de chômage des jeunes** (`YouthUnemployment`) dans différentes zones géographiques et pour des années futures.

### Type d'Analyse

Le problème est catégorisé comme un problème de **Régression** :

* **Variable Cible (Target) :** `YouthUnemployment` (Taux, valeur numérique continue).
* **Modèle Attendu :** Un modèle de régression (par exemple, Régression Linéaire, Modèle ARIMA pour séries temporelles, ou un algorithme d'apprentissage automatique comme Random Forest Regressor ou XGBoost) sera entraîné pour estimer la valeur de ce taux.

---

## 4. Dictionnaire des Données et Métadonnées

Le jeu de données se compose de **quatre (4) variables** principales.

### Taille et Structure

| Métadonnée | Valeur |
| :--- | :--- |
| **Format** | CSV |
| **Unités d'Observation** | Une ligne représente le taux de chômage des jeunes pour un pays/région spécifique à une année donnée. |
| **Features (Variables Explicatives)** | `Country`, `CountryCode`, `Year` |
| **Target (Variable Cible)** | `YouthUnemployment` |

### Description des Variables

| Nom de la Colonne | Type de Donnée | Description | Rôle |
| :--- | :--- | :--- | :--- |
| **Country** | `Object` (Catégorielle Nominale) | Nom complet du pays ou de la région agrégée (ex: "France", "Euro Area"). | Feature |
| **CountryCode** | `Object` (Catégorielle Nominale) | Code alphanumérique (généralement ISO 3166-1 alpha-2 ou code de groupe) identifiant le pays/la région. | Feature |
| **Year** | `Int64` (Numérique Discrète) | Année de la mesure. Cruciale pour l'analyse des séries temporelles. | Feature |
| **YouthUnemployment** | `Float64` (Numérique Continue) | Taux de chômage des jeunes (en pourcentage). | **Target (Cible)** |

---

## 5. Étapes Suivantes

La prochaine phase du projet se concentrera sur les points suivants :

1.  **Nettoyage des Données :** Gestion des valeurs manquantes (`NaN`), notamment dans la colonne `YouthUnemployment`.
2.  **Analyse Exploratoire des Données (EDA) :**
    * Visualisation de l'évolution du taux moyen global.
    * Identification des pays/régions avec les taux les plus élevés/faibles.
    * Analyse de la distribution de la variable cible.
3.  **Ingénierie des Caractéristiques (Feature Engineering) :** Création de variables dérivées (ex: indicateurs de tendance ou de variation annuelle) et encodage des variables catégorielles (`Country`, `CountryCode`).
4.  **Modélisation :** Sélection et entraînement des modèles de régression.
5.  **Évaluation :** Utilisation de métriques appropriées (ex: RMSE, MAE, $R^2$) pour évaluer la performance du modèle.
