# Parameters
param n; 				# rows 
param m; 				# columns (features)
param nu;				

param y_tr {1..m};     
param A_tr {1..m,1..n};   

param y_te {1..m};        
param A_te {1..m,1..n};   

# Variables
var w {1..n};
var gamma;             
var s {1..m};       


# Primal formulation
minimize svm_primal:
	(1/2)*sum{j in {1..n}}(w[j]^2) +nu*sum{i in {1..m}}(s[i]);
	
subject to c1 {i in {1..m}}:
	y_tr[i]*(sum{j in {1..n}}(A_tr[i,j]*w[j]) + gamma) >=  + 1 -s[i];

subject to c2 {i in {1..m}}:
	s[i] >= 0;
	
	
		