# Auteur: Esteban Debordes
# Projet: Hackathon Tectonic - Moteur Prédictif KBC (Version Corrégée & Complète)

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter


def charger_et_reconstituer_historique(chemin_fichier=r"C:\Users\Esteb\Downloads\Donnétest.json"):
    """
    Lit le fichier JSON agrégé, sépare les charges récurrentes mensuelles des
    achats exceptionnels/ponctuels, et reconstruit l'historique du solde.
    """
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        donnees = json.load(f)
        
    transactions_historiques = []
    calendrier_fixe = []

    for item in donnees:
        dates_parsed = [datetime.fromisoformat(d) for d in item["dates_apparition"]]
        is_fixe = item.get("charge_fixe", False) or item.get("recette", False)
        categorie = item.get("categorie", "Inconnu")
        
        # --- RÈGLE DE RÉCURRENCE MENSUELLE ---
        # Un flux est récurrent mensuel SEULEMENT s'il apparaît au moins 3 fois
        # et n'est pas une catégorie ponctuelle (Éducation, Exceptionnel)
        est_recurrent_mensuel = (len(dates_parsed) >= 3) and (categorie not in ["Éducation", "Exceptionnel"])
        
        for d in dates_parsed:
            transactions_historiques.append({
                "Date": d, 
                "Montant": item["montant"],
                "Is_Fixe": is_fixe,
                "Categorie": categorie
            })
            
        # Ajout au calendrier prédictif uniquement si c'est une vraie charge/recette récurrente
        if is_fixe and est_recurrent_mensuel:
            jours_du_mois = [d.day for d in dates_parsed]
            jour_typique = Counter(jours_du_mois).most_common(1)[0][0]
            
            calendrier_fixe.append({
                "nom": item["marchand"],
                "jour_typique": jour_typique,
                "montant": item["montant"]
            })

    df_historique = pd.DataFrame(transactions_historiques)
    df_historique = df_historique.sort_values(by="Date").reset_index(drop=True)
    df_historique["Solde"] = df_historique["Montant"].cumsum()

    date_fin = df_historique["Date"].max()

    return df_historique, date_fin, calendrier_fixe


def calculer_statistiques_variables(df_historique, demi_vie_jours=30):
    """
    Calcule mu et sigma des dépenses quotidiennes variables en utilisant une 
    pondération temporelle exponentielle (EWMA).
    """
    categories_exclues = ["Éducation", "Exceptionnel"]
    mask_variable = (~df_historique["Is_Fixe"]) & (~df_historique["Categorie"].isin(categories_exclues))
    df_var = df_historique[mask_variable].copy()
    
    if df_var.empty:
        return -5.0, 8.0, -8.0, 12.0

    # Grille temporelle quotidienne (Zero-Filling des jours sans dépense)
    s_journaliere = df_var.set_index("Date").resample("D")["Montant"].sum().fillna(0)
    
    # 1. Calcul des poids exponentiels sous forme de tableau NumPy
    date_max = s_journaliere.index.max()
    jours_ecoules = (date_max - s_journaliere.index).days.to_numpy()
    alpha = np.log(2) / demi_vie_jours
    poids = pd.Series(np.exp(-alpha * jours_ecoules), index=s_journaliere.index)
    
    # 2. Séparation Semaine / Week-end
    est_weekend = s_journaliere.index.weekday >= 5
    
    # 3. Moyenne et écart-type pondérés (sécurisés avec np.asarray)
    def calculer_stats_ponderees(series, weights):
        s_arr = np.asarray(series)
        w_arr = np.asarray(weights)
        if len(s_arr) == 0 or w_arr.sum() == 0:
            return -5.0, 8.0
        mu_w = np.average(s_arr, weights=w_arr)
        var_w = np.average((s_arr - mu_w)**2, weights=w_arr)
        sigma_w = np.sqrt(var_w)
        return mu_w, sigma_w

    mu_semaine, sigma_semaine = calculer_stats_ponderees(s_journaliere[~est_weekend], poids[~est_weekend])
    mu_weekend, sigma_weekend = calculer_stats_ponderees(s_journaliere[est_weekend], poids[est_weekend])
    
    # Sécurités de bornage
    sigma_semaine = 8.0 if sigma_semaine == 0 or pd.isna(sigma_semaine) else sigma_semaine
    sigma_weekend = 12.0 if sigma_weekend == 0 or pd.isna(sigma_weekend) else sigma_weekend

    return mu_semaine, sigma_semaine, mu_weekend, sigma_weekend


def simuler_monte_carlo_avancee(df_historique, solde_depart, date_depart, calendrier_fixe,
                                 jours_projection=35, n_simulations=1000, 
                                 lambda_imprevu=0.015, montant_imprevu_moy=-60, std_imprevu=15):
    """
    Exécute la simulation Monte Carlo avec bruit quotidien, charges fixes et imprévus.
    """
    mu_sem, sigma_sem, mu_wk, sigma_wk = calculer_statistiques_variables(df_historique)

    dates_futures = [date_depart + timedelta(days=i) for i in range(1, jours_projection + 1)]
    simulations = np.zeros((n_simulations, jours_projection))
    
    for sim in range(n_simulations):
        solde_courant = solde_depart
        for t, date in enumerate(dates_futures):
            is_weekend = date.weekday() >= 5
            mu = mu_wk if is_weekend else mu_sem
            sigma = sigma_wk if is_weekend else sigma_sem
            
            # 1. Bruit quotidien normal
            choc_gaussien = np.random.normal(mu, sigma)
            choc_gaussien = min(0.0, choc_gaussien)  # Une dépense variable ne crédite pas le compte
            
            # 2. Imprévu financier occasionnel (Loi de Poisson)
            choc_imprevu = 0.0
            if np.random.poisson(lambda_imprevu) > 0:
                choc_imprevu = np.random.normal(montant_imprevu_moy, std_imprevu)
            
            # 3. Application des flux fixes récurrents prévus à cette date
            flux_fixes = sum(f["montant"] for f in calendrier_fixe if date.day == f["jour_typique"])
            
            solde_courant += choc_gaussien + choc_imprevu + flux_fixes
            simulations[sim, t] = solde_courant

    # Calcul des quantiles et de la probabilité de découvert
    q10 = np.percentile(simulations, 10, axis=0)
    q50 = np.percentile(simulations, 50, axis=0)
    q90 = np.percentile(simulations, 90, axis=0)
    prob_decouvert = np.mean(simulations < 0, axis=0) * 100

    df_resultats = pd.DataFrame({
        "Date": dates_futures,
        "P10_Pessimiste": q10,
        "P50_Median": q50,
        "P90_Optimiste": q90,
        "Probabilite_Decouvert_%": prob_decouvert
    })
    
    return df_resultats, simulations


def afficher_graphique_monte_carlo(df_historique, df_res, simulations, max_trajectoires=40):
    """Affiche le graphique de prédiction et la jauge de risque."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Affichage de l'historique récent (60 derniers jours)
    date_limite = df_historique["Date"].max() - timedelta(days=60)
    df_hist_recent = df_historique[df_historique["Date"] >= date_limite]
    
    # --- GRAPHIQUE 1 : SOLDE & PROJECTIONS ---
    ax1.plot(df_hist_recent["Date"], df_hist_recent["Solde"], color="#00529B", linewidth=2.5, label="Historique Réel")
    
    for i in range(min(max_trajectoires, len(simulations))):
        ax1.plot(df_res["Date"], simulations[i], color="#E3000F", alpha=0.05, linewidth=0.8)
        
    ax1.plot(df_res["Date"], df_res["P50_Median"], color="#E3000F", linewidth=2.2, label="Médiane (p50)")
    ax1.fill_between(df_res["Date"], df_res["P10_Pessimiste"], df_res["P90_Optimiste"], 
                     color="#E3000F", alpha=0.15, label="Cône d'Incertitude (p10-p90)")
    
    ax1.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Solde (€)")
    ax1.set_title("Modèle Prédictif KBC - Simulation Monte Carlo (Données Corrigées)", fontweight='bold', fontsize=13)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # --- GRAPHIQUE 2 : RISQUE DE DÉCOUVERT ---
    ax2.bar(df_res["Date"], df_res["Probabilite_Decouvert_%"], color="#F58220", alpha=0.8, width=0.8)
    ax2.axhline(50, color="red", linestyle=":", label="Seuil Critique (50%)")
    ax2.set_ylabel("Risque Découvert (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    chemin_json = r"C:\Users\Esteb\Downloads\Donnétest.json"
    
    print("🔄 Traitement des données financières...")
    df_historique, date_derniere_transaction, charges_fixes = charger_et_reconstituer_historique(chemin_json)
    
    # Ajustement de démo : Positionner le solde à 150 € au 30 septembre
    solde_brut = df_historique["Solde"].iloc[-1]
    solde_cible_demo = 150.00
    df_historique["Solde"] = df_historique["Solde"] + (solde_cible_demo - solde_brut)
    solde_actuel = solde_cible_demo
    
    print(f"\n📊 --- STATUT DU COMPTE ---")
    print(f"Solde de départ au {date_derniere_transaction.strftime('%d/%m/%Y')} : {solde_actuel:.2f} €")
    print(f"Abonnements / Charges récurrentes identifiées : {len(charges_fixes)}")

    print(f"\n🎲 Calcul des prédictions (1 000 trajectoires)...")
    df_res, simulations = simuler_monte_carlo_avancee(
        df_historique=df_historique,
        solde_depart=solde_actuel,
        date_depart=date_derniere_transaction,
        calendrier_fixe=charges_fixes,
        jours_projection=35,
        n_simulations=1000
    )
    
    # Analyse d'alerte
    jour_critique = df_res[df_res["Probabilite_Decouvert_%"] >= 50]
    print(f"\n🚨 --- DÉTECTION DE RISQUE ---")
    if not jour_critique.empty:
        d_alerte = jour_critique.iloc[0]["Date"].strftime("%d/%m/%Y")
        p_val = jour_critique.iloc[0]["Probabilite_Decouvert_%"]
        print(f"⚠️  ALERTE PRÉVENTIVE : Risque de découvert de {p_val:.1f}% à partir du {d_alerte}.")
    else:
        print("✅ Aucun risque de découvert majeur détecté sur les 35 prochains jours.")

    print("\n📈 Génération de la visualisation...")
    afficher_graphique_monte_carlo(df_historique, df_res, simulations)