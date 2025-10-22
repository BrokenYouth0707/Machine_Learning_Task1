import numpy as np

#Chose z-score normalization to transform the data points

# Original 20 data points
a = np.array([27,26,20,31,14,16,25,7,6,10,15,31,6,28,26,16,28,24,11,24], dtype=float)
b = np.array([442,442,457,431,471,463,434,487,480,480,470,430,489,436,444,464,447,430,475,445], dtype=float)

def compute_z_scores(data, ddof=1):
    mean = data.mean()
    std  = data.std(ddof=ddof)
    return (data - mean) / std, mean, std

# Export standardized data + stats (population SD, ddof=1)
x, x_mean, x_std = compute_z_scores(a, ddof=1)
y, y_mean, y_std = compute_z_scores(b, ddof=1)

if __name__ == "__main__":
    print(x.tolist())
    print(y.tolist())
