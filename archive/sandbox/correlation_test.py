from neuralpredictors.measures.np_functions import corr
import numpy as np

a = np.array([[2, 5], [3, 5], [4, 6], [2, 6], [2, 6]])
b = np.array([[2, 5], [2, 5], [3, 5], [2, 5], [2, 6]])
corr(a, b, axis=0) # should be equal to [0.87499997, 0.40824827]
np.mean(corr(a, b, axis=0))