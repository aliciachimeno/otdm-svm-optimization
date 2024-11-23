# Parameters
param m; 				# rows
param n; 				# columns
param nu;				# tradeoff
param A_train {1..n,1..m};   	# feature values
param y_train {1..n};        	# response value
param A_test {1..n,1..m};   	# feature values
param y_test {1..n};        	# response value

# Variables
var w {1..m};
var gamma;             	
var s {1..n};          	


# Primal problem formulation
minimize svm_primal:
	(1/2)*sum{j in {1..m}}(w[j]^2) +nu*sum{i in {1..n}}(s[i]);
	
subject to c1 {i in {1..n}}:
	-y_train[i]*(sum{j in {1..m}}(A_train[i,j]*w[j]) + gamma) -s[i] + 1 <= 0;

subject to c2 {i in {1..n}}:
	-s[i] <= 0;
