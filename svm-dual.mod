# Parameters:
param m; 				# rows
param n; 				# columns
param nu;				# tradeoff
param A_train {1..n,1..m};   	# feature values
param y_train {1..n};        	# response value
param A_test {1..n,1..m};   	# feature values
param y_test {1..n};        	# response value

# Variables:
var lambda {1..n} >= 0, <= nu;

# Dual formulation
maximize svm_dual:
	sum{i in {1..n}}lambda[i] 
	-(1/2)*sum{i in {1..n}, j in {1..n}}lambda[i]*y_train[i]*lambda[j]*y_train[j]*(sum{k in {1..m}}A_train[i,k]*A_train[j,k]);
		
subject to c1:
	sum{i in {1..n}}(lambda[i]*y_train[i]) = 0;