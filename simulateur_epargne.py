"""
Simulateur d'Objectif d'Épargne - Version Ultra-Simplifiée
"""

from dataclasses import dataclass

@dataclass
class FluxMensuel:
    nom: str
    montant: float
    recette: bool
    charge_fixe: bool

def simuler_objectif_epargne(flux_mensuels: list[FluxMensuel], objectif_epargne: float, horizon_mois: int = 12):
    print(f"--- ANALYSE DE L'OBJECTIF : ÉPARGNER {objectif_epargne}€/MOIS ---")
    
    # ---------------------------------------------------------
    # a. Calcul de la capacité d'épargne
    # ---------------------------------------------------------
    revenus = sum(f.montant for f in flux_mensuels if f.recette)
    depenses = sum(f.montant for f in flux_mensuels if not f.recette)
    surplus_actuel = revenus - depenses
    
    print(f"\na. Capacité d'épargne actuelle : {surplus_actuel} € / mois")
    print(f"   (Revenus: {revenus}€ - Dépenses: {depenses}€)")

    # ---------------------------------------------------------
    # b. Analyse de compressibilité
    # ---------------------------------------------------------
    fixes = sum(f.montant for f in flux_mensuels if not f.recette and f.charge_fixe)
    flexibles = sum(f.montant for f in flux_mensuels if not f.recette and not f.charge_fixe)
    
    print(f"\nb. Compressibilité des dépenses :")
    print(f"   - Incompressibles (Fixes) : {fixes} €")
    print(f"   - Flexibles (Loisirs, etc.) : {flexibles} €")

    # ---------------------------------------------------------
    # e. Garde-fous éthiques (Seuil décent)
    # ---------------------------------------------------------
    # On fixe arbitrairement qu'une personne doit garder au moins 400€ pour vivre (nourriture, santé)
    RESTE_A_VIVRE_MIN = 400 
    reste_a_vivre_max_possible = revenus - fixes
    
    print(f"\ne. Garde-fou éthique :")
    print(f"   Le Reste-à-Vivre absolu après charges fixes est de {reste_a_vivre_max_possible}€.")
    print(f"   Seuil minimum vital fixé à {RESTE_A_VIVRE_MIN}€.")

    # ---------------------------------------------------------
    # c. Scénarios d'ajustement
    # ---------------------------------------------------------
    print(f"\nc. Scénario d'ajustement :")
    nouveau_surplus = surplus_actuel
    effort_a_trouver = objectif_epargne - surplus_actuel

    if effort_a_trouver <= 0:
        print("   ✅ Objectif déjà atteint avec votre rythme actuel !")
        nouveau_surplus = surplus_actuel
    else:
        # On regarde si on peut couper dans le flexible sans tuer le niveau de vie
        epargne_max_ethique = reste_a_vivre_max_possible - RESTE_A_VIVRE_MIN
        
        if objectif_epargne > epargne_max_ethique:
            print("   ❌ DANGER : Atteindre cet objectif vous mettrait sous le seuil de pauvreté.")
            print(f"   Plan adapté : On plafonne l'épargne à {epargne_max_ethique}€/mois.")
            nouveau_surplus = epargne_max_ethique
        else:
            reduction_pct = (effort_a_trouver / flexibles) * 100
            print(f"   ⚠️ Plan d'action : Réduisez vos dépenses flexibles de {reduction_pct:.0f}%")
            print(f"   (Trouver {effort_a_trouver}€ d'économies dans les {flexibles}€ de dépenses flexibles).")
            nouveau_surplus = objectif_epargne

    # ---------------------------------------------------------
    # d. Projection impactée
    # ---------------------------------------------------------
    print(f"\nd. Projection sur {horizon_mois} mois :")
    print("Mois | Épargne Sans Effort | Épargne Avec Plan")
    print("-" * 50)
    
    cumul_actuel = 0
    cumul_plan = 0
    
    for mois in range(1, horizon_mois + 1):
        cumul_actuel += surplus_actuel
        cumul_plan += nouveau_surplus
        print(f" {mois:02d}  | {cumul_actuel:17.2f} € | {cumul_plan:15.2f} €")


# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
if __name__ == "__main__":
    mes_flux_mensuels = [
        FluxMensuel(nom="Salaire", montant=2200, recette=True, charge_fixe=True),
        FluxMensuel(nom="Loyer", montant=800, recette=False, charge_fixe=True),
        FluxMensuel(nom="Électricité", montant=100, recette=False, charge_fixe=True),
        FluxMensuel(nom="Assurances", montant=50, recette=False, charge_fixe=True),
        # Dépenses flexibles
        FluxMensuel(nom="Courses", montant=400, recette=False, charge_fixe=False),
        FluxMensuel(nom="Restaurants", montant=250, recette=False, charge_fixe=False),
        FluxMensuel(nom="Shopping", montant=200, recette=False, charge_fixe=False),
    ]

    # Test avec un objectif réaliste (ex: 300€/mois)
    simuler_objectif_epargne(mes_flux_mensuels, objectif_epargne=300, horizon_mois=6)
    
    print("\n\n" + "="*50 + "\n\n")
    
    # Test avec un objectif déraisonnable (ex: 1200€/mois)
    simuler_objectif_epargne(mes_flux_mensuels, objectif_epargne=1200, horizon_mois=6)