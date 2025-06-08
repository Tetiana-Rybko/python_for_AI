import pandas as pd
from sklearn.model_selection import train_test_split


data = {
    "text": [
        "This movie was fantastic!",
        "Terrible film, I hated it.",
        "Loved the acting, very impressive.",
        "Worst movie I’ve seen.",
        "Brilliant story and great performance.",
        "I do not recommend this film.",
        "Amazing direction and strong emotions.",
        "Not my type, boring scenes.",
        "Superb! Will watch again.",
        "Awful, waste of time."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

df.drop_duplicates(inplace=True)
df["text"] = df["text"].str.lower()


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Данные подготовлены: train.csv и test.csv")