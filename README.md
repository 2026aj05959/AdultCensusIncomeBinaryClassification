# Hotel Booking Cancellation Classification

## a. Problem statement
Hotel cancellations can make it harder for hotels to plan room availability, staffing, and expected revenue. This project uses the Kaggle Hotel Booking Demand dataset to examine whether information recorded about a booking can help predict if it will later be canceled. The dataset contains booking details such as hotel type, lead time, arrival information, guest counts, room type, deposit type, and special requests. Kaggle dataset

This is a supervised binary classification problem. The input variables are the booking details, and the target variable is is_canceled: 1 means the booking was canceled; 0 means it was not canceled. The project trains and compares Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine models on the same training and test split. It evaluates them using accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient.

The goal is to identify which model best distinguishes canceled bookings from non-canceled bookings on held-out data. The fields reservation_status and reservation_status_date are excluded because they contain information about the booking’s final outcome and could let a model infer the answer directly. The resulting model comparison is for this dataset and academic exercise; its performance should not be assumed to generalize to every hotel or future booking pattern.

## b. Dataset description
- Source: [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- The dataset  `hotel_bookings.csv` is downloaded from Kaggle and placed it at `data/hotel_bookings.csv` beside `train.py`.
- The dataset contains 119,390 rows and 32 columns, including the target.
- `is_canceled` is the target. `reservation_status` and `reservation_status_date` are excluded because they reveal information recorded after the cancellation outcome and would leak the answer.
- The script uses a stratified 10,000-row sample for manageable training, then makes a stratified 80/20 training/test split.

## c. Github link
 https://github.com/2026aj05959/HotelBookingCancellationModelEvaluation/tree/main

## d. Comparison Table with the evaluation metrics calculated for all the 5 models

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8100 | 0.8941 | 0.7282 | 0.7772 | 0.7519 | 0.5990 |
| Decision Tree | 0.8293 | 0.8213 | 0.7624 | 0.7832 | 0.7727 | 0.6362 |
| KNN | 0.8088 | 0.8764 | 0.7527 | 0.7204 | 0.7362 | 0.5867 |
| Naive Bayes | 0.4669 | 0.5777 | 0.4082 | 0.9767 | 0.5758 | 0.2185 |
| Support Vector Machine | 0.8352 | 0.9096 | 0.7689 | 0.7935 | 0.7810 | 0.6491 |

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 0.8100 accuracy and an AUC of 0.8941, indicating good ability to distinguish canceled from non-canceled bookings. Its recall for canceled bookings was 0.7772. The confusion matrix shows 12,831 false positives and 9,854 false negatives. |
| Decision Tree | Achieved 0.8293 accuracy, 0.7624 precision, 0.7832 recall, an F1 score of 0.7727, and an MCC of 0.6362. It performs well overall, though its AUC of 0.8213 is lower than Logistic Regression, KNN, and SVM. Its confusion matrix shows 10,791 false positives and 9,589 false negatives. |
| KNN | Achieved 0.8088 accuracy and an AUC of 0.8764. Its recall of 0.7204 is lower than the other models except for no model in this comparison, meaning it misses more canceled bookings. The confusion matrix shows 10,466 false positives and 12,365 false negatives. |
| Naive Bayes | Achieved very high recall for canceled bookings (0.9767), missing only 1,032 cancellations. However, it incorrectly classified 62,610 non-canceled bookings as canceled. Its accuracy (0.4669), precision (0.4082), AUC (0.5777), and MCC (0.2185) are low, showing many false cancellation alerts and weak overall performance. |
| Support Vector Machine | Achieved the best results across all six comparison metrics: accuracy (0.8352), AUC (0.9096), precision (0.7689), recall (0.7935), F1 (0.7810), and MCC (0.6491). Its confusion matrix shows 10,547 false positives and 9,133 false negatives. |
| Overall Winner for This Dataset | Support Vector Machine. It has the highest value for each of the six reported metrics and provides the strongest overall balance when identifying canceled bookings while limiting false cancellation predictions. |

