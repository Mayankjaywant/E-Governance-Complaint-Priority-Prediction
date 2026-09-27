import joblib
import pandas as pd

model = joblib.load("complaint_priority_model.joblib")

new_complaint = pd.DataFrame([{
    "category": "Water",
    "days_pending": 12,
    "severity": 5,
    "previous_complaints": 2,
    "citizen_age": 34,
    "distance_km": 4.5,
    "is_essential_service": 1,
    "repeat_issue": 1
}])

prediction = model.predict(new_complaint)[0]
print("Predicted complaint priority:", prediction)
