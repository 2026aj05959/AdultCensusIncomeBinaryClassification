# Hotel Booking Cancellation Classification

## Problem statement
Predict whether a hotel booking will be canceled (`is_canceled = 1`) or not canceled (`is_canceled = 0`) from booking attributes. This is a binary classification exercise.

## Dataset description
- Source: [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Download `hotel_bookings.csv` from Kaggle and place it at `data/hotel_bookings.csv` beside `train.py`.
- The dataset contains 119,390 rows and 32 columns, including the target; it therefore meets the assignment minimum of 500 instances and 12 features.
- `is_canceled` is the target. `reservation_status` and `reservation_status_date` are excluded because they reveal information recorded after the cancellation outcome and would leak the answer.
- The script uses a stratified 10,000-row sample for manageable training, then makes a stratified 80/20 training/test split.
