# Uber Request Data Analysis: Supply-Demand Gap

## Objective
Identify when and where Uber fails to fulfil ride requests (cancellations and
"no cars available") and recommend ways to close the supply-demand gap.

## Dataset
Kaggle "Uber Request Data" schema: Request id, Pickup point (City/Airport),
Driver id, Status, Request timestamp, Drop timestamp.
> Place the real `Uber Request Data.csv` in this folder before running.
> If it is absent, the script generates a synthetic dataset with the same schema.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn

## Method
1. Data cleaning: mixed date formats, duplicates, expected nulls
2. Feature engineering: hour, time slot, trip duration, failure flag
3. EDA: status mix, hourly demand, status by hour and slot, trip duration
4. Gap analysis: demand minus completed trips per pickup point and slot
5. Insights and recommendations

## Key Findings (on bundled synthetic data; re-check with real data)
- ~60% of requests completed; ~40% fail.
- City, early morning: highest cancellations.
- Airport, evening rush: highest "No Cars Available", the largest gap.

## Recommendations
- Driver incentives for morning city-to-airport trips
- Surge pricing and driver pre-positioning at the airport in the evening
- Airport queue/holding-lot alerts; penalties for repeat cancellations

## Run
```
pip install pandas numpy matplotlib seaborn
python uber_analysis.py
```
Outputs: `charts/`, `cleaned_uber_data.csv`, `insights.txt`
