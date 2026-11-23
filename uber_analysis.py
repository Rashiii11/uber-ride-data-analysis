"""
Uber Request Data Analysis: Supply-Demand Gap
----------------------------------------------
Goal: find when and where Uber fails to fulfil requests
(cancellations / no cars available) and recommend fixes.

Input : Uber Request Data.csv  (Kaggle schema). If missing, a synthetic
        dataset with the same schema is generated so the project runs.
Output: charts/*.png, cleaned_uber_data.csv, insights.txt
"""
# %% 1. Imports
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)
RAW = "Uber Request Data.csv"

# %% 2. Load (or generate) data
def generate_data(n=6500, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.date_range("2016-07-11", periods=5)
    city_w = np.array([1,1,1,1,3,8,9,8,5,3,2,2,2,2,2,3,3,4,4,3,3,2,2,1], float)
    air_w = np.array([2,1,1,1,1,2,2,2,2,2,2,2,2,2,3,4,6,8,9,8,6,4,3,2], float)
    for i in range(n):
        point = rng.choice(["City", "Airport"])
        w = city_w if point == "City" else air_w
        hour = rng.choice(24, p=w / w.sum())
        ts = rng.choice(days) + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(60)),
                                              seconds=int(rng.integers(60)))
        # Failure patterns: city mornings -> cancellations, airport evenings -> no cars
        if point == "City" and 5 <= hour <= 9:
            p = [0.42, 0.45, 0.13]
        elif point == "Airport" and 17 <= hour <= 21:
            p = [0.25, 0.10, 0.65]
        else:
            p = [0.80, 0.10, 0.10]
        status = rng.choice(["Trip Completed", "Cancelled", "No Cars Available"], p=p)
        driver = np.nan if status == "No Cars Available" else int(rng.integers(1, 301))
        drop = ts + pd.Timedelta(minutes=int(rng.normal(52, 12))) if status == "Trip Completed" else pd.NaT
        rows.append([i + 1, point, driver, status, ts, drop])
    df = pd.DataFrame(rows, columns=["Request id", "Pickup point", "Driver id", "Status",
                                     "Request timestamp", "Drop timestamp"])
    # mimic messy real-world formats
    fmt = lambda s: s.dt.strftime("%d-%m-%Y %H:%M:%S") if True else s
    half = df.index % 2 == 0
    df["Request timestamp"] = df["Request timestamp"].astype(object)
    df.loc[half, "Request timestamp"] = pd.to_datetime(df.loc[half, "Request timestamp"]).dt.strftime("%d/%m/%Y %H:%M")
    df.loc[~half, "Request timestamp"] = pd.to_datetime(df.loc[~half, "Request timestamp"]).dt.strftime("%d-%m-%Y %H:%M:%S")
    df["Drop timestamp"] = df["Drop timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
    return df

if os.path.exists(RAW):
    df = pd.read_csv(RAW)
    print("Loaded real dataset:", df.shape)
else:
    df = generate_data()
    df.to_csv(RAW, index=False)
    print("Real dataset not found -> generated synthetic data:", df.shape)

# %% 3. Cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates(subset="Request id")
for c in ["Request timestamp", "Drop timestamp"]:
    df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce", format="mixed")
df["Pickup point"] = df["Pickup point"].str.strip().str.title()
df["Status"] = df["Status"].str.strip()

print("\nMissing values:\n", df.isna().sum())
# Missing Driver id = 'No Cars Available'; missing Drop time = trip not completed (expected)

# %% 4. Feature engineering
df["Hour"] = df["Request timestamp"].dt.hour
df["Day"] = df["Request timestamp"].dt.date

def slot(h):
    if 4 <= h < 9:   return "Early Morning (4-9)"
    if 9 <= h < 12:  return "Late Morning (9-12)"
    if 12 <= h < 17: return "Afternoon (12-17)"
    if 17 <= h < 22: return "Evening Rush (17-22)"
    return "Night (22-4)"
df["Time slot"] = df["Hour"].apply(slot)
df["Trip minutes"] = (df["Drop timestamp"] - df["Request timestamp"]).dt.total_seconds() / 60
order = ["Early Morning (4-9)", "Late Morning (9-12)", "Afternoon (12-17)",
         "Evening Rush (17-22)", "Night (22-4)"]
palette = {"Trip Completed": "#2ca02c", "Cancelled": "#ff7f0e", "No Cars Available": "#d62728"}

# %% 5. EDA
overall = df["Status"].value_counts()
print("\nOverall status:\n", overall, "\n", (overall / len(df) * 100).round(1))

# Chart 1: overall status
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Status", hue="Status", palette=palette, order=overall.index, legend=False)
plt.title("Request Status Overview"); plt.tight_layout()
plt.savefig("charts/1_status_overview.png", dpi=150); plt.close()

# Chart 2: hourly demand by pickup point
plt.figure(figsize=(10, 4.5))
sns.countplot(data=df, x="Hour", hue="Pickup point")
plt.title("Hourly Request Demand by Pickup Point"); plt.tight_layout()
plt.savefig("charts/2_hourly_demand.png", dpi=150); plt.close()

# Chart 3: status by hour
plt.figure(figsize=(11, 4.5))
sns.countplot(data=df, x="Hour", hue="Status", palette=palette)
plt.title("Request Status by Hour"); plt.tight_layout()
plt.savefig("charts/3_status_by_hour.png", dpi=150); plt.close()

# Chart 4: status by time slot and pickup point
g = sns.catplot(data=df, x="Time slot", hue="Status", col="Pickup point", kind="count",
                order=order, palette=palette, height=4.5, aspect=1.2)
g.set_xticklabels(rotation=30, ha="right"); g.fig.suptitle("Status by Time Slot", y=1.03)
g.savefig("charts/4_slot_by_pickup.png", dpi=150); plt.close("all")

# Chart 5: trip duration
plt.figure(figsize=(6, 4))
sns.histplot(df["Trip minutes"].dropna(), bins=25, kde=True, color="#1f77b4")
plt.title("Trip Duration Distribution (min)"); plt.tight_layout()
plt.savefig("charts/5_trip_duration.png", dpi=150); plt.close()

# %% 6. Problem identification: where is the gap?
df["Failed"] = df["Status"] != "Trip Completed"
tbl = pd.crosstab([df["Pickup point"], df["Time slot"]], df["Status"])
tbl["Total"] = tbl.sum(axis=1)
tbl["Fail %"] = ((tbl["Total"] - tbl["Trip Completed"]) / tbl["Total"] * 100).round(1)
tbl = tbl.reindex(order, level=1)
print("\nStatus by pickup point & time slot:\n", tbl)

# Chart 6: heatmap of failure rate
heat = df.pivot_table(index="Time slot", columns="Pickup point", values="Failed", aggfunc="mean").reindex(order) * 100
plt.figure(figsize=(6, 4))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "% failed"})
plt.title("Failure Rate (%) by Slot and Pickup"); plt.tight_layout()
plt.savefig("charts/6_failure_heatmap.png", dpi=150); plt.close()

# Supply-demand gap = requests - completed trips
gap = df.groupby(["Pickup point", "Time slot"]).agg(
    demand=("Request id", "count"), supply=("Status", lambda s: (s == "Trip Completed").sum()))
gap["gap"] = gap["demand"] - gap["supply"]
gap = gap.sort_values("gap", ascending=False)
print("\nTop supply-demand gaps:\n", gap.head(4))

# %% 7. Insights
worst_cancel = df[df.Status == "Cancelled"].groupby(["Pickup point", "Time slot"]).size().idxmax()
worst_nocar = df[df.Status == "No Cars Available"].groupby(["Pickup point", "Time slot"]).size().idxmax()
top = gap.index[0]
completed_pct = overall.get("Trip Completed", 0) / len(df) * 100

insights = f"""UBER SUPPLY-DEMAND GAP: KEY INSIGHTS
====================================
Total requests analysed : {len(df):,}
Trips completed         : {completed_pct:.1f}%
Cancelled               : {overall.get('Cancelled',0)/len(df)*100:.1f}%
No cars available       : {overall.get('No Cars Available',0)/len(df)*100:.1f}%
Avg trip duration       : {df['Trip minutes'].mean():.0f} min

1. Most cancellations: {worst_cancel[0]} pickups in {worst_cancel[1]}.
   Drivers likely cancel city->airport trips (long, low return trip demand).
2. Most 'No Cars Available': {worst_nocar[0]} pickups in {worst_nocar[1]}.
   Demand from arriving passengers exceeds cars waiting at the airport.
3. Largest absolute gap: {top[0]} / {top[1]} ({int(gap.iloc[0]['gap'])} unfulfilled requests).

RECOMMENDATIONS
- Incentivise drivers (bonus / guaranteed fare) for city->airport morning trips to cut cancellations.
- Surge pricing + pre-positioning of drivers near the airport during evening rush.
- Airport holding-lot / queue notifications so idle drivers know when demand spikes.
- Show riders honest ETAs and penalise repeat driver cancellations.
"""
print("\n" + insights)
open("insights.txt", "w").write(insights)
df.to_csv("cleaned_uber_data.csv", index=False)
