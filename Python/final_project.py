import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Import the CPI DataFrame and compute the inflation DataFrame
filename = "CPI.csv"
df_cpi = pd.read_csv(filename)
df_cpi.Date = pd.to_datetime(df_cpi['Date'])
df_cpi = df_cpi.set_index('Date')
df_inflation = (df_cpi - df_cpi.shift(12)) / df_cpi.shift(12) * 100

# Plot the preliminary data
fig, ax = plt.subplots(3, 1, sharey = False)
ax[0].plot(df_inflation.index, df_inflation["Personal Consumption"], color = 'black', linestyle = '-', label = "Headline Inflation")
ax[0].plot(df_inflation.index, df_inflation["Gasoline and Energy"], color = 'gray', linestyle = '--', label = "Gasoline and Energy")
ax[0].set_ylabel("Inflation (percentage)")
ax[0].legend(loc = 'upper left', frameon = False)

ax[1].plot(df_inflation.index, df_inflation["Personal Consumption"], color = 'black', linestyle = '-', label = "Headline Inflation")
ax[1].plot(df_inflation.index, df_inflation["Housing and Utilities"], color = 'gray', linestyle = '--', label = "Housing and Utilities")
ax[1].set_ylabel("Inflation (percentage)")
ax[1].legend(loc = 'upper left', frameon = False)

ax[2].plot(df_inflation.index, df_inflation["Personal Consumption"], color = 'black', linestyle = '-', label = "Headline Inflation")
ax[2].plot(df_inflation.index, df_inflation["Health Care"], color = 'gray', linestyle = '--', label = "Health Care")
ax[2].set_ylabel("Inflation (percentage)")
ax[2].set_xlabel("Time")
ax[2].legend(loc = 'upper left', frameon = False)
fig.set_size_inches([15, 8])
fig.savefig("Times_Series.png")

# Subsetting the time series data for only the great moderation period
df = df_inflation.loc[df_inflation.index > pd.to_datetime('1959-12-01')]
df.to_csv('inflation_data.csv')
# Create the X and y matrix
X = df.values[:-12, :]
y = df.shift(-12)['Personal Consumption'].values.reshape([-1, 1])[:-12, :]
y = y.reshape(-1)
# Illustrate time series cross validation 
cv = TimeSeriesSplit(n_splits = 564, max_train_size = 120)
fig, ax = plt.subplots(figsize = (10 ,5))
for ii, (tr, tt) in enumerate(cv.split(X, y)):
	if df.index[tt] < pd.to_datetime('2005-01-01'):
		l1 = ax.scatter(df.index[tr], [ii] * len(tr), c = [plt.cm.binary(.5)], marker = "_", lw = 0.1)
		l2 = ax.scatter(df.index[tt + 12], [ii] * len(tt), c = [plt.cm.binary(.9)], marker = "_", lw = 1)
	else:
		l3 = ax.scatter(df.index[tr], [ii] * len(tr), c = [plt.cm.coolwarm(.1)], marker = "_", lw = 0.1)
		l4 = ax.scatter(df.index[tt + 12], [ii] * len(tt), c = [plt.cm.coolwarm(.9)], marker = "_", lw = 1)
	ax.set(ylim=[564, -1], xlabel = "Data Index", ylabel = "CV Iteration")
	ax.legend([l1, l2], ["Training", "Validation"])
fig.set_size_inches([15, 4])
fig.savefig("TimeSeriesSplit.png")

# Define the function that returns 
def compute_RMSE(model, cv, X, y, heldout = False):
	y_predictions = []
	y_truth = []
	time_index = []
	coefs = []
	for tr, tt in cv.split(X, y):
		if not heldout:
			if df.index[tt + 12] < pd.to_datetime('2005-01-01'):
				model.fit(X[tr], y[tr])
				#coefs.append(model.coef_)
				y_predictions.append(model.predict(X[tt]))
				y_truth.append(y[tt])
				time_index.append(df.index[tt + 12])
		else:
			if df.index[tt + 12] >= pd.to_datetime('2005-01-01'):
				model.fit(X[tr], y[tr])
				#coefs.append(model.coef_)
				y_predictions.append(model.predict(X[tt]))
				y_truth.append(y[tt])
				time_index.append(df.index[tt + 12])
	y_truth = np.array(y_truth).reshape(-1,1)
	y_predictions = np.array(y_predictions).reshape(-1,1)
	time_index = np.array(time_index).reshape(-1,1)
	RMSE = mean_squared_error(y_truth, y_predictions, squared = True)
	return RMSE, y_truth, y_predictions, time_index

# The following three lines of code compute the RMSE for three benchmark models
#print(compute_RMSE(Lasso(alpha = 0), cv, X, y, heldout = True)[0])
#print(compute_RMSE(Ridge(alpha = 0), cv, X, y, heldout = True)[0])
#print(compute_RMSE(LinearRegression(), cv, X[:, 0].reshape(-1, 1), y)[0])

# Search in the hyperparameter space for Ridge
#svr_c_vec = np.linspace(50, 100, 25)
#svr_RMSE_vec = np.array([compute_RMSE(SVR('rbf', C = c), cv, X, y)[0] for c in svr_c_vec])
#svr_gamma_vec = np.linspace(0.001, 0.01, 10)
#svr_gamma_RMSE_vec = np.array([compute_RMSE(SVR('rbf', C = 77, gamma = gamma), cv, X, y)[0] for gamma in svr_gamma_vec])

#fig, ax = plt.subplots(1, 2)
#ax[0].plot(svr_gamma_vec, svr_gamma_RMSE_vec, marker = "o", linestyle = "--", linewidth = 0.5, markersize = 0.5, color = 'black')
#ax[0].set_ylabel("RMSE")
#ax[0].set_xlabel("gamma")

#ax[1].plot(svr_c_vec, svr_RMSE_vec, marker = "o", linestyle = "--", linewidth = 0.5, markersize = 0.5, color = 'black')
#ax[1].set_ylabel("RMSE")
#ax[1].set_xlabel("C")
#fig.set_size_inches([15, 4])
#fig.savefig("Tuning_SVR.png")

r_RMSE, r_y_truth, r_y_predictions, r_time_index = compute_RMSE(Ridge(alpha = 0.), cv, X, y)
s_RMSE, s_y_truth, s_y_predictions, s_time_index = compute_RMSE(SVR('rbf', C = 77, gamma = 0.0035), cv, X, y)
r_RMSE2, r_y_truth2, r_y_predictions2, r_time_index2 = compute_RMSE(Ridge(alpha = 0.), cv, X, y, heldout = True)
s_RMSE2, s_y_truth2, s_y_predictions2, s_time_index2 = compute_RMSE(SVR('rbf', C = 77, gamma = 0.0035), cv, X, y, heldout = True)

fig, ax = plt.subplots(2, 1)
ax[0].plot(r_time_index2, r_y_truth2, marker = "o", markersize = 0.3, color = 'black', linestyle = "-", label = "Actual Inflation")
ax[0].plot(r_time_index2, r_y_predictions2, marker = "+", color = 'blue', linestyle = "--", label = "Ridge Forecast")
ax[0].set_ylabel("Inflation")
ax[0].set_xlabel("Time")
ax[0].legend(loc = 'lower left', frameon = False)

ax[1].plot(s_time_index2, s_y_truth2, marker = "o", markersize = 0.3, color = 'black', linestyle = "-", label = "Actual Inflation")
ax[1].plot(s_time_index2, s_y_predictions2, marker = "+", color = 'blue', linestyle = "--", label = "SVR Forecast")
ax[1].set_ylabel("Inflation")
ax[1].set_xlabel("Time")
ax[1].legend(loc = 'lower left', frameon = False)
fig.set_size_inches([15, 8])
fig.savefig("Forecast_HO.png")
#print(s_RMSE)
#print(s_RMSE2)




