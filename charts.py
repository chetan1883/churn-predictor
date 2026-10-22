import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean
df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# Churn rate (%) for each category of a column
def churn_rate_chart(column, title):
    rate = (
        df.groupby(column)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    print(f"\n{title}\n{rate.round(1)}")

    ax = rate.plot(kind="bar", color="#d9534f", figsize=(7, 4))
    ax.set_title(title)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"charts/{column}.png")
    plt.close()

churn_rate_chart("Contract", "Churn rate by contract type")
churn_rate_chart("InternetService", "Churn rate by internet service")
churn_rate_chart("PaymentMethod", "Churn rate by payment method")
churn_rate_chart("TechSupport", "Churn rate by tech support")

# Tenure: how long customers stay before leaving
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack")
plt.title("Customer tenure (months) by churn")
plt.tight_layout()
plt.savefig("charts/tenure.png")
plt.close()

# Monthly charges: do churners pay more?
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly charges by churn")
plt.tight_layout()
plt.savefig("charts/monthly_charges.png")
plt.close()

print("\nDone! Charts saved in the 'charts' folder.")