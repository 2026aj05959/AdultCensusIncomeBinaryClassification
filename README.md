# Hotel Booking Cancellation Classification

## Problem statement
Hotel cancellations can make it harder for hotels to plan room availability, staffing, and expected revenue. This project uses the Kaggle Hotel Booking Demand dataset to examine whether information recorded about a booking can help predict if it will later be canceled. The dataset contains booking details such as hotel type, lead time, arrival information, guest counts, room type, deposit type, and special requests. Kaggle dataset

This is a supervised binary classification problem. The input variables are the booking details, and the target variable is is_canceled: 1 means the booking was canceled; 0 means it was not canceled. The project trains and compares Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine models on the same training and test split. It evaluates them using accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient.

The goal is to identify which model best distinguishes canceled bookings from non-canceled bookings on held-out data. The fields reservation_status and reservation_status_date are excluded because they contain information about the booking’s final outcome and could let a model infer the answer directly. The resulting model comparison is for this dataset and academic exercise; its performance should not be assumed to generalize to every hotel or future booking pattern.

## Dataset description
- Source: [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Download `hotel_bookings.csv` from Kaggle and place it at `data/hotel_bookings.csv` beside `train.py`.
- The dataset contains 119,390 rows and 32 columns, including the target; it therefore meets the assignment minimum of 500 instances and 12 features.
- `is_canceled` is the target. `reservation_status` and `reservation_status_date` are excluded because they reveal information recorded after the cancellation outcome and would leak the answer.
- The script uses a stratified 10,000-row sample for manageable training, then makes a stratified 80/20 training/test split.
