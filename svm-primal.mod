# Parameters
param n; 				# rows
param m; 				# columns
param nu;				# tradeoff

param y_tr {1..m};        	# response value
param A_tr {1..m,1..n};   	# feature values

param y_te {1..m};        	# response value
param A_te {1..m,1..n};   	# feature values

# Variables
var w {1..n};
var gamma;             	# intercept
var s {1..m};          	# slack


# Primal formulation
minimize svm_primal:
	(1/2)*sum{j in {1..n}}(w[j]^2) +nu*sum{i in {1..m}}(s[i]);
	
subject to c1 {i in {1..m}}:
	-y_tr[i]*(sum{j in {1..n}}(A_tr[i,j]*w[j]) + gamma) -s[i] + 1 <= 0;

subject to c2 {i in {1..m}}:
	-s[i] <= 0;