for the digit dataset run:

        python main_digit.py
        
        
the program prints the training time, number of misclassified samples and the accuracy score for all the requested classifiers.


### EEG Eye State Dataset
This dataset describes EEG data for an individual and whether their eyes were open or closed. The objective of the problem is to predict whether eyes are open or closed given EEG data alone.
This is a classification predictive modeling problems and there are a total of 14,980 observations and 15 input variables. The class value of ‘1’ indicates the eye-closed and ‘0’ the eye-open state. Data is ordered by time and observations were recorded over a period of 117 seconds.
 This dataset is from UCI Machine Learning Repository, and is available from this URL: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
 
 for the EEG Eye state dataset dataset run:

        python main_timeseries.py
        
the program prints the training time, number of misclassified samples and the accuracy score for all the requested classifiers.

### Some sklearn decision tree pruning strategies 
* Max-depth: The maximum depth of the tree. defined in tree.py line 86.
* Min-samples-split: The minimum number of samples required to split an internal node. defined in tree.py line 100

