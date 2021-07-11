# Dimensional Reduction Analysis
In order to create a robust model, strongly correlated dimensions should be 
eliminated such that all dimensions represent indenpendent components.

One way to visualize this step is to create a covariance matrix which calculates
how to variables "co-vary". If two variables increase together their covariance
would be positive. If they had no _linear_ relationship the covariance will be
close to zero.

Different models are anticipated for red wine and white wine. Futhermore, the 
dimensional reduction analysis is also performed seperately.

## Red Wine Dim-Reduction
The dim_reduce.py script produced the following result
![title](RedWineCovariance.png)
It is important to note that the variables were normalized
before the taking their covariance. This would not allow variables
with a larger dynamic range to create a fictious correlation.

The following variables are proposed to be removed
* free sulfur dioxide
* citric acid
* fixed acidity
* Volatile acidity
* chlorides
* Density (both alcohol and pH have strong correlations)

I am curious to see if the pH can cover the other three measures of acidity in
the model.

These changes produce the following covariance matrix
![](RedWineCovariance_adj.png)
