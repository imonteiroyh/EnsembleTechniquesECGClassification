# Lobachevsky University Electrocardiogram Arrhythmia Classification Study
Arrhythmia detection and classification study using ensemble techniques on Lobachevsky University Eletrocardiography Database. Study carried out for the Digital Signal Processing Laboratory Course in December 2022.

# Access the files
- Download the ZIP file in [Physionet](https://physionet.org/content/ludb/1.0.1/)
- Download the files using the terminal: ```wget -r -N -c -np https://physionet.org/content/ludb/1.0.1/```

# Method
![ludb_method](https://user-images.githubusercontent.com/61994795/206858469-dccbfa24-c60f-4bb6-b75d-5519f894ada6.png)

# Getting started
1. Get the data files
2. Run script.sh via ```bash script.sh```
3. Run ecg_rhythm_classifier.ipynb

# About Bagging and Boosting
Bagging and Boosting are two types of Ensemble techniques, where ensemble is a Machine Learning concept in which the idea is to train multiple models using the same learning algorithm. The Bagging and Boosting models get the number of learners by producing new training data sets using random sampling with replacement, in other words, some observations may be repeated in each new training data set. In the case of Bagging, any element has the same probability of appearing in a new dataset whilst for Boosting the observations are weighted. The final decision to classify the samples is made with an arithmetic mean for Bagging while it is made with a weighted arithmetic mean for Boosting.

![bagging_and_boosting](https://user-images.githubusercontent.com/61994795/206860120-7dc405fd-c4e2-451a-9759-1f610aed2cc5.png)
