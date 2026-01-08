from src.data.main import build
from src.eda.main import explore
from src.models.main import model

if __name__ == "__main__":
    df = None
    #df = build()
    df = explore(df)
    model(df)

# TODO: Goal for today - finish EDA!
    # DONE: Write about PCA
    # DONE: Write about ANOVA

# TODO: Goal for tomorrow - finish modelling
    # DONE: Add the relevant PCS as features
        # PC related features to add:
        # country-profile PC2 feature: country_development_efficiency = PC2
    # DONE: test PCs in correlation 
    # DONE: Retrain baseline model with new features
    # TODO: Train tree-based decision models including XGBoost
    # TODO: Write interpretation

# TODO: Goal for Friday - finish
    # TODO: Write everything
    # TODO: Maybe add nice visualizations, like map


