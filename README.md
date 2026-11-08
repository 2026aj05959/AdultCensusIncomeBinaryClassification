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
| Logistic Regression | 0.8090 | 0.8923 | 0.7292 | 0.7706 | 0.7493 | 0.5958 |
| Decision Tree | 0.8110 | 0.8000 | 0.7410 | 0.7530 | 0.7470 | 0.5962 |
| KNN | 0.7870 | 0.8565 | 0.7247 | 0.6856 | 0.7046 | 0.5387 |
| Naive Bayes | 0.4670 | 0.5773 | 0.4085 | 0.9784 | 0.5763 | 0.2207 |
| Support Vector Machine | 0.8320 | 0.9105 | 0.7725 | 0.7746 | 0.7736 | 0.6400 |

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Performs well overall, with a strong AUC of 0.8923. Its recall (0.7706) is close to the SVM's, so it identifies canceled bookings at a similar rate. Its accuracy and precision are lower than the Decision Tree's and SVM's. |
| Decision Tree | Achieves 0.8110 accuracy, slightly higher than Logistic Regression and KNN. Its MCC (0.5962) is almost the same as Logistic Regression's. Its AUC (0.8000) is lower, indicating weaker ability to distinguish the two classes across thresholds. |
| KNN | Has a moderate AUC (0.8565), but lower accuracy (0.7870), recall (0.6856), and F1 (0.7046) than Logistic Regression, Decision Tree, and SVM. It misses more canceled bookings than these three models. |
| Naive Bayes | Has the highest recall (0.9784), identifying nearly all canceled bookings. However, its accuracy (0.4670), precision (0.4085), AUC (0.5773), and MCC (0.2207) are low. Its high recall comes with many false cancellation predictions. |
| Support Vector Machine | Gives the strongest overall results: it has the highest accuracy (0.8320), AUC (0.9105), precision (0.7725), F1 (0.7736), and MCC (0.6400). Its recall (0.7746) is also strong, though lower than Naive Bayes's. |
| Overall Winner for this Dataset | Support Vector Machine. It leads on five of the six reported metrics and offers the best balance of precision, recall, and overall classification performance. If the priority were to catch nearly every canceled booking, Naive Bayes has the highest recall, but its many false positives make it weaker overall. |

