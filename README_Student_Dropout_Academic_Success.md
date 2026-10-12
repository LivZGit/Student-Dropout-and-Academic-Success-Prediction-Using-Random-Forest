# Student Dropout and Academic Success Prediction Using Random Forest

## Overview

This project uses Python and machine learning to analyse student data
and predict one of three academic outcomes:

-   **Dropout**
-   **Enrolled**
-   **Graduate**

It was developed as **PBL Activity 3 for Python for Data Science
(BE05000231)**. The notebook includes data inspection and cleaning
checks, exploratory data analysis (EDA), statistical analysis,
visualizations, feature association analysis, Random Forest training,
and model evaluation.

## Objectives

-   Explore the Student Dropout and Academic Success dataset.
-   Examine student outcome distributions and selected feature
    relationships.
-   Analyse quantitative-feature correlations and use mutual information
    to rank associations with the target.
-   Train a Random Forest classifier to predict student outcomes.
-   Evaluate the classifier using accuracy, precision, recall, F1-score,
    and a confusion matrix.
-   Review feature importance and demonstrate a sample prediction.

## Dataset

The notebook expects a file named `data.csv` in the same working
directory and loads it with a semicolon delimiter:

``` python
df = pd.read_csv("data.csv", sep=";")
```

The dataset contains **4,424 records**, **36 input features**, and a
target column named `Target`. The target classes are Dropout, Enrolled,
and Graduate.

The dataset is from the UCI Machine Learning Repository:

[Predict Students' Dropout and Academic
Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

Please follow the dataset's terms and citation requirements when
redistributing it. If you do not include the dataset in this repository,
download it from the official source and place the required CSV file in
the notebook's working directory.

## Methodology

1.  Load the dataset and inspect its structure.
2.  Check target labels, missing values, and duplicate records.
3.  Perform EDA using summaries and charts.
4.  Explore quantitative-feature correlations.
5.  Use mutual information to rank the association between selected
    input features and student outcome.
6.  Remove the first- and second-semester curricular-unit performance
    features, leaving **24 input features**.
7.  Encode the target labels and split the data into training and
    testing sets using an **80:20 stratified split**.
8.  Train a `RandomForestClassifier` with **100 trees** and
    `random_state=42`.
9.  Evaluate the model and inspect feature importance and a sample
    prediction.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   SciPy
-   Scikit-learn
-   Jupyter Notebook / Google Colab

## Results

The notebook reports the following test-set results:

-   **Accuracy:** 63.39%
-   **Dropout recall:** 62%
-   **Enrolled recall:** 15%
-   **Graduate recall:** 82%

Performance varies by class. In particular, the model has low recall for
the Enrolled class, so it should be treated as an educational project
and not used on its own to make high-stakes decisions about individual
students.

## Running the Notebook

1.  Download or clone this repository.

2.  Ensure the required `data.csv` file is available in the notebook's
    working directory.

3.  Open the `.ipynb` notebook in Google Colab or Jupyter Notebook.

4.  Install any missing dependencies, if required:

    ``` bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    ```

5.  Run the notebook cells in order.

> **Note:** The notebook loads `data.csv` using a relative path. In
> Google Colab, upload the CSV to the session or update the data-loading
> path to match where you store the file.

## Repository Contents

-   `Student_Dropout_Academic_Success_Random_Forest_Using_Python.ipynb`
    --- project notebook.
-   `README.md` --- project overview and instructions.
-   `data.csv` --- required input dataset, if you choose and are
    permitted to include it.

## Limitations and Future Work

-   The model's performance is uneven across the three outcome classes.
-   Enrolled-class recall is low and needs improvement.
-   Future work could compare additional classifiers, tune
    hyperparameters with cross-validation, investigate class imbalance,
    and validate the model on suitable current data.
-   Feature association and model importance indicate relationships or
    predictive usefulness; they do not establish causation.

## Author

**Ishwar Devnarayan Vishwakarma**\
B.E. Computer Engineering --- Third Year, Semester V\
C. K. Pithawala College of Engineering and Technology

## Academic Context

Python for Data Science (BE05000231) --- PBL Activity 3.
