'''
    Emma Gershman
    DS2500
    Homework 5
'''

# import libraries

import pandas as pd
from PIL.ImagePalette import random
from nltk.parse.generate import demo_grammar
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.ticker import FuncFormatter

# declare and initialize filename constants

FILENAME_PRESIDENTS = "1976-2020-president.tab"
FILENAME_DEMOGRAPHICS = "demographics_HW5.csv"

def clean_demographics(df):
    '''
    Parameters: dataframe for demographics
    Does: formats dataframe as specified
    Returns: formatted/cleaned dataframe
    '''

    df["STNAME"] = df["STNAME"].str.upper()

    df["PERCENT_MALE"] = df["TOT_MALE"] / df["TOT_POP"]
    df["PERCENT_FEMALE"] = df["TOT_FEMALE"] / df["TOT_POP"]
    df["PERCENT_WHITE"] = (df["WA_MALE"] + df["WA_FEMALE"]) / df["TOT_POP"]
    df["PERCENT_BLACK"] = df["Black"] / df["TOT_POP"]
    df["PERCENT_HISPANIC"] = df["Hispanic"] / df["TOT_POP"]

    df.rename(columns={"STNAME": "state"}, inplace=True)

    df = df[
        ["state", "PERCENT_MALE", "PERCENT_FEMALE", "PERCENT_WHITE",
         "PERCENT_BLACK", "PERCENT_HISPANIC"]]
    return df

def clean_presidents(df):
    '''
    Parameters: presidential dataframe
    Does: drops unnecessary columns and cleans presidential data
    Returns: clean dataframe
    '''
    df.drop(
        ["state_po", "state_fips", "state_cen", "state_ic", "office", "candidate", "writein", "version", "notes",
         "party_simplified"], axis=1, inplace=True)
    df["party_detailed"] = df["party_detailed"].replace({"DEMOCRATIC-FARMER-LABOR": "DEMOCRAT"})
    return df

def keep_winning_party(df):
    '''
    Parameters: presidental dataframe
    Does: determines winning party for each state per year
    Returns: filtered dataframe
    '''

    df = df.sort_values(by="candidatevotes", ascending=False)

    df = df.drop_duplicates(subset=["state", "year"])

    df.replace("DEMOCRAT", 0, inplace=True)
    df.replace("REPUBLICAN", 1, inplace=True)
    return df

def create_knn_model(merged_df, factors, label, k):
    '''
    Parameters: merged dataframe, factors, label, k-value
    Does: creates KNN object and trains model
    Returns: dictionary containing X_train set,
                                    X_test set,
                                    y_train set,
                                    y_test set,
                                    y_pred,
                                    knn object
    '''
    X = merged_df[factors]
    y = merged_df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    dct = {"X_train" : X_train,
           "X_test" : X_test,
           "y_train" : y_train,
           "y_test" : y_test,
           "y_pred" : y_pred,
           "knn" : knn}
    return dct

def count_correct_predict(y_pred, y_test):
    '''
    Parameters: y_pred, y_test
    Does: counts how many times the model correctly predicted the winning party
    Returns: count
    '''
    counter = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            counter += 1
    return counter

def evaluate_k(merged_df, factors, label, start_k, end_k):
    '''
    Parameters: merged dataframe, factors, label, start k-value, end k-value
    Does: determines most accurate k value, most accurate score, and the
          maximum number of correct predictions
    Returns: dictionary containing all 3 calculated values
    '''
    best_accuracy_k = start_k
    best_accuracy_score = 0
    max_num_correct = 0
    for i in range(start_k, end_k + 1):
        temp_dct = create_knn_model(merged_df, factors, label, i)
        temp_accuracy_score = accuracy_score(temp_dct["y_test"],
                                             temp_dct["y_pred"])
        if temp_accuracy_score > best_accuracy_score:
            best_accuracy_score = temp_accuracy_score
            best_accuracy_k = i
        temp_correct = count_correct_predict(temp_dct["y_pred"],
                                             temp_dct["y_test"].tolist())
        if temp_correct > max_num_correct:
            max_num_correct = temp_correct
    ret_dct = {"best k" : best_accuracy_k, "max correct" : max_num_correct}
    return ret_dct

def create_confusion_matrix(merged_df, factors, label, best_k):
    '''
    Parameters: merged dataframe, factors, label, best k-value
    Does: creates a heatmap showing the confusion matrix when
          the value of k is optimal (out of 5, 6, ... 10) for accuracy.
    Returns: nothing
    '''

    knn_stats = create_knn_model(merged_df, factors, label, best_k)

    cm = confusion_matrix(knn_stats["y_test"].tolist(), knn_stats["y_pred"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix Heatmap When K = {best_k}")
    plt.show()

def percent_formatter(x, pos):
    '''
    Parameters: x value
    Does: formats x value
    Returns: formatted x value
    '''
    return f'{x * 100:.0f}%'

def create_scatterplot(merged_df, year, k):
    '''
    Parameters: merged dataframe, user-chosen year, random k value
    Does: creates a scatterplot using percent white and percent hispanic
          features, with different colors and/or shapes for: training set
           (voted republican), training set (voted democrat), testing set.
    Returns: nothing
    '''

    factors = ["PERCENT_WHITE", "PERCENT_HISPANIC"]
    label = "party_detailed"

    knn_stats = create_knn_model(merged_df, factors, label, k)

    X_train = knn_stats["X_train"]  # Shape: (38, 2)
    y_train = knn_stats["y_train"]  # Shape: (38,)

    training_data_df = pd.DataFrame(X_train,
                                    columns=["PERCENT_WHITE",
                                             "PERCENT_HISPANIC"])
    training_data_df["target"] = y_train


    testing_data_df = pd.DataFrame(knn_stats["X_test"],
                                   columns = ["PERCENT_WHITE",
                                              "PERCENT_HISPANIC"])
    testing_data_df["target"] = knn_stats["y_test"]

    x_dems = training_data_df.drop("PERCENT_HISPANIC", axis = 1)
    x_dems = x_dems[x_dems["target"] == 0]

    x_reps = training_data_df.drop("PERCENT_HISPANIC", axis = 1)
    x_reps = x_reps[x_reps["target"] == 1]

    y_dems = training_data_df.drop("PERCENT_WHITE", axis = 1)
    y_dems = y_dems[y_dems["target"] == 0]

    y_reps = training_data_df.drop("PERCENT_WHITE", axis = 1)
    y_reps = y_reps[y_reps["target"] == 1]

    x_testing = testing_data_df["PERCENT_WHITE"]
    y_testing = testing_data_df["PERCENT_HISPANIC"]

    plt.scatter(x_dems, y_dems, marker = "o", color = "blue",
                label = "Democrat")
    plt.scatter(x_reps, y_reps, marker = "o", color = "red",
                label = "Republican")
    plt.scatter(x_testing, y_testing, marker = "x", color = "black",
                label = "Testing Set")

    formatter = FuncFormatter(percent_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel("Percent White")
    plt.ylabel("Percent Hispanic")
    plt.legend()
    plt.title(f"{year} Voting Prediction By Demographics")
    plt.show()

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # read in files as dataframes
    presidents = pd.read_csv(FILENAME_PRESIDENTS, sep = "\t")
    demographics = pd.read_csv(FILENAME_DEMOGRAPHICS)

    # clean data
    demographics = clean_demographics(demographics)
    presidents = clean_presidents(presidents)

    # keep only winning party
    presidents = keep_winning_party(presidents)

    # sort dataframe by states in alphabetical order
    presidents = presidents.sort_values(by = "state", ascending = True)

    # prompt user for a year they want to build
    year = int(input("Enter an election year from 1976 through 2020: "))

    presidents_filtered = presidents[presidents["year"] == year]
    presidents_filtered = presidents_filtered[["state", "party_detailed"]]

    merged_df = pd.merge(presidents_filtered, demographics, on = "state")
    factors = ["PERCENT_MALE", "PERCENT_FEMALE", "PERCENT_WHITE", "PERCENT_BLACK", "PERCENT_HISPANIC"]
    label = "party_detailed"
    k_val = int(input("Enter a k value: "))
    dct_k = create_knn_model(merged_df, factors, label, k_val)

    correct_predict_k = count_correct_predict(dct_k["y_pred"], dct_k["y_test"].tolist())
    print(f"2. When k = {k_val}, {correct_predict_k} states in the testing set are predicted correctly.")

    # calculate f1 score for democrats
    report = classification_report(dct_k["y_test"].tolist(), dct_k["y_pred"], output_dict = True)
    f1_dems = round(report["0"]["f1-score"], 2)
    print(f"3. The F1 score for the states that voted democrat is {f1_dems}.")

    # prompt user for state
    state = input("Enter state: ").upper()

    given_row = merged_df.loc[merged_df["state"] == state, ["PERCENT_MALE", "PERCENT_FEMALE", "PERCENT_WHITE", "PERCENT_BLACK", "PERCENT_HISPANIC"]]

    # predict winning party for given state in given year
    user_predict = dct_k["knn"].predict(given_row)
    user_actual = merged_df.loc[merged_df["state"] == state, "party_detailed"].tolist()
    print(f"predicted: {user_predict}")
    print(f"actual: {user_actual}")

    if user_predict == user_actual:
        print(f"4. Yes, my model correctly predicted how {state} voted in {year}!")
    else:
        print(f"4. No, my model did not correctly predict how {state} voted in {year}.")

    # determine best k value in range 5-10 (inclusive)
    START_K = 5
    END_K = 10
    k_stats = evaluate_k(merged_df, factors, label, START_K, END_K)
    print(f"5. The best accuracy is when k = {k_stats["best k"]}.")
    print(f"   The maximum number of states that my model can predict correctly is {k_stats["max correct"]}.")

    # create heatmap
    create_confusion_matrix(merged_df, factors, label, k_stats["best k"])

    # create scatterplot
    random_k = random.randint(5, 10)
    create_scatterplot(merged_df, year, random_k)

main()