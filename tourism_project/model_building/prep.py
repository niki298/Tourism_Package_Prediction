import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Remove the saved row index and customer identifier
df = df.drop(columns=["Unnamed: 0", "CustomerID"])

# Standardize the spelling of Female
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Separate input features (X) from the target (y)
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Reserve 20% for testing, preserving the purchase proportions
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save the four datasets without adding a CSV row index
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Report the split sizes and target proportions
print("Data preparation completed.")
print(f"Training features: {Xtrain.shape}")
print(f"Test features: {Xtest.shape}")

print("\nTraining target counts:")
print(ytrain.value_counts())

print("\nTest target counts:")
print(ytest.value_counts())

print("\nPurchase proportions:")
print(f"Training: {ytrain.mean():.2%}")
print(f"Test: {ytest.mean():.2%}")
