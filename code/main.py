from src.data.main import build
from src.eda.main import explore
from src.models.main import model

if __name__ == "__main__":
    df = build()
    explore(df)
    #model(df)

# TODO: Goal for today - finish EDA!
    # DONE: Write about PCA
    # DONE: Write about ANOVA

# TODO: Goal for tomorrow - finish modelling
    # TODO: Add the relevant PCS as features
        # PC related features to add:
        # country-profile PC2 feature: country_development_efficiency = PC2
        # high_volatility_product = 1 if PC2 < 0 else 0 (products)
        # product_globalization_index = PC1
    # TODO: test political dummies + PCs in correlation 
    # TODO: Retrain baseline model with new features
    # TODO: Train tree-based decision models including XGBoost
    # TODO: Write interpretation

# TODO: Goal for Friday - finish
    # TODO: Write everything
    # TODO: Maybe add nice visualizations, like map


