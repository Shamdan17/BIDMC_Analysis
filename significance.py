
# This file contains functions for calculating statistical significance
# for different filter & detection pairs 
# The aim is to validate the significance of filters and detection methods 
from scipy.stats import ttest_ind

def apply_significance_test(predictions, true_values, confidence_level=95):
    """
    Apply statistical significance test to a given set of predictions and true values
    :param predictions: list of predictions
    :param true_values: list of true values
    :param alpha: significance level
    :return: True if the null hypothesis is rejected, False otherwise
    """

    alpha = 1 - confidence_level / 100

    is_significant = ttest_ind(predictions, true_values)[1] < alpha
    
    return is_significant
