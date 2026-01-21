import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Define main folder
main_folder = r"C:\Users\user\Desktop\Projects\EchoFeeling\data\sticker"

# 2. Collect counts per category
counts = {}
for subfolder in os.listdir(main_folder):
    path = os.path.join(main_folder, subfolder)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(".png")])
        counts[subfolder] = count

# 3. Convert to DataFrame for easy handling
df = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])

# 4. Print summary
print("Sticker dataset distribution:")
print(df)

print(f"\nTotal PNG images: {df['Count'].sum()}")

# 5. Plot distribution
colors = {"positive":"green", "negative":"red", "neutral":"blue"}
df.plot(kind="bar", x="Category", y="Count", legend=False,
        color=[colors.get(cat, "gray") for cat in df["Category"]])
plt.title("Sticker Dataset Distribution")
plt.ylabel("Number of Images")
plt.show()
