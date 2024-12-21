# Analysis of Differential Privacy Papers

## Orchard: Differentially Private Analytics at Scale

* Implemented sequential laplace noise mechanism on data pull followed by the bmcs aggregation implementation - broadcasts state to devices, maps local to global private data, and clips elements per device, aggregating the private vectors.
* Evaluating impact of varying sensitivity and epsilon parameters, generating noisy means from repeated runs of bmcs and visualizing mean noisy sums by sensitivity and epsilon parameters.

## Improved Differentially Private Regression via Gradient Boosting

* Examining noisy gradient descent and performing sensitivity analysis for the perturbations on properties of the hessian and hat matrix.

## Differentially-Private “Draw and Discard” Machine Learning

* Python implementation of differentially private stochastic gradient descent for logistic regression.
