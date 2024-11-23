# Parameters:
param n; 				
param m; 		
param nu;              	

param y_tr {1..m};        	
param A_tr {1..m,1..n};   	

param y_te {1..m};        	# response value
param A_te {1..m,1..n};   	# feature values

# Variables
var lambda {1..m} >= 0, <= nu;


# Dual formulation
maximize svm_dual:
	sum{i in {1..m}}lambda[i] 
	-(1/2)*sum{i in {1..m}, j in {1..m}}lambda[i]*y_tr[i]*lambda[j]*y_tr[j]*(sum{k in {1..n}}A_tr[i,k]*A_tr[j,k]);
	
subject to c1:
	sum{i in {1..m}}(lambda[i]*y_tr[i]) = 0;