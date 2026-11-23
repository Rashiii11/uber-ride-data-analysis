Uber Ride Request Analysis: Supply-Demand Gap

Problem Statement

Uber loses revenue and rider trust when ride requests fail. Requests from the
City and the Airport often end as Cancelled or No Cars Available.
This project finds when and where these failures happen and why.

Solution

I analysed ride request data in Python to:

	•	Clean and standardise mixed timestamp formats
	•	Create features: hour of day, time slot, trip duration, failure flag
	•	Compare demand vs completed trips (the supply-demand gap) by pickup point and time slot
	•	Recommend ways to reduce failed requests

Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn

Dataset

Uses the Kaggle "Uber Request Data" schema (Request id, Pickup point, Driver id,
Status, Request timestamp, Drop timestamp). The data in this repo is a
synthetic sample with the same schema, so results are illustrative.
The script also works with the real Kaggle file.

Screenshots

Failure rate by time slot and pickup point

Request status by hour

Key Findings

	•	About 60% of requests complete; about 40% fail.
	•	City pickups in the early morning (4-9) have the most cancellations.
	•	Airport pickups in the evening rush (17-22) have the most "No Cars Available" and the largest gap.

Recommendations

	•	Give drivers incentives for morning city-to-airport trips
	•	Use surge pricing and pre-position drivers at the airport in the evening
	•	Send airport queue / holding-lot alerts to idle drivers
	•	Penalise repeat driver cancellations

Setup Steps

	1.	Install Python 3.9+
	2.	pip install pandas numpy matplotlib seaborn
	3.	Download uber_analysis.py
	4.	(Optional) Put the real Uber Request Data.csv in the same folder
	5.	Run python uber_analysis.py

Outputs: charts, cleaned_uber_data.csv, insights.txt

Live Link

Run it online: ADD_YOUR_GOOGLE_COLAB_LINK_HERE

Conclusion

Failures are concentrated in two windows: city mornings (driver cancellations)
and airport evenings (not enough cars). Targeted driver incentives and
pricing in these windows would close most of the gap.

Author

Rashiii11