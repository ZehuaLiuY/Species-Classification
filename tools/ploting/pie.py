import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Class": [
        "american black bear", "american marten", "american red squirrel",
        "black-tailed jackrabbit", "bobcat", "california ground squirrel",
        "california quail", "cougar", "coyote", "dark-eyed junco",
        "domestic cow", "domestic dog", "donkey", "dusky grouse",
        "eastern gray squirrel", "elk", "ermine", "european badger",
        "gray fox", "gray jay", "horse", "house wren", "long-tailed weasel",
        "moose", "mule deer", "nine-banded armadillo",
        "north american porcupine", "north american river otter",
        "raccoon", "red deer", "red fox", "snowshoe hare",
        "steller's jay", "striped skunk", "unidentified accipitrid",
        "unidentified bird", "unidentified chipmunk", "unidentified corvus",
        "unidentified deer", "unidentified deer mouse", "unidentified mouse",
        "unidentified pack rat", "unidentified pocket gopher",
        "unidentified rabbit", "virginia opossum", "wild boar",
        "wild turkey", "yellow-bellied marmot"
    ],
    "Prevalence": [
        2962, 142, 330, 105, 2560, 3277, 365, 1639, 2215, 2, 15555, 187, 319, 0,
        2918, 2420, 0, 16, 1034, 9, 28, 0, 5, 1166, 10706, 924, 48, 73, 3611,
        15015, 170, 1389, 0, 1200, 30, 9156, 137, 117, 10793, 17, 0, 72, 0, 436,
        136, 11519, 469, 42
    ]
}

df = pd.DataFrame(data)

df_nonzero = df[df["Prevalence"] > 0]

df_top = df_nonzero.sort_values("Prevalence", ascending=False).head(15)
df_other = pd.DataFrame({
    "Class": ["Other"],
    "Prevalence": [df_nonzero["Prevalence"].sum() - df_top["Prevalence"].sum()]
})
df_pie = pd.concat([df_top, df_other], ignore_index=True)

plt.figure(figsize=(10, 10))
plt.pie(df_pie["Prevalence"], labels=df_pie["Class"], autopct='%1.1f%%', startangle=140)
plt.title("Class Distribution in the Test Set (Top 15 Classes + Other)")
plt.tight_layout()
plt.savefig('class_distribution_pie.pdf', dpi=600)
plt.show()