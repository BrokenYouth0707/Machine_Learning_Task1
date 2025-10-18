import numpy as np

# Chose z-score normalization to transform the data points

# Original 20 data points a = x and b = y
a = np.array([27, 26, 20, 31, 14, 16, 25, 7, 6, 10, 15, 31, 6, 28, 26, 16, 28, 24, 11, 24])
b = np.array([442, 442, 457, 431, 471, 463, 434, 487, 480, 480, 470, 430, 489, 436, 444, 464, 447, 430, 475, 445])

def compute_z_scores(data, ddof=0):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=ddof)
    z_scores = (data - mean) / std_dev
    return z_scores

# Initialize empty lists for z-scores. These will hold the standardized/tranformed data points
x_arr = compute_z_scores(a, ddof = 1)
y_arr = compute_z_scores(b, ddof = 1)
x_list = x_arr.tolist()
y_list = y_arr.tolist()

print(x_list)
print(y_list)

