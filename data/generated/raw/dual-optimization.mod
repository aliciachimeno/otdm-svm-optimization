reset;
print "SVM_DUAL:";
model svm-dual.mod;
data "./data/generated/ampl_format/data1000_converted.dat";
#data "./data/real/ampl_format/real_dataset.dat"
option solver cplex;
problem SVM_DUAL: lambda, svm_dual, c1;
solve SVM_DUAL;
display lambda;


## get w and gamma

param w {1..n};
let {j in {1..n}} w[j] := sum{i in {1..m}} lambda[i]*y_tr[i]*A_tr[i,j];
display w;

param gamma;
for {i in {1..m}} {
	if lambda[i] > 0.01 and lambda[i] < nu*0.99 then {
		# A support vector point was found
		let gamma := 1/y_tr[i] - sum{j in {1..n}} w[j]*A_tr[i,j];
		break;
	}
}
display gamma;


# Compare solve with the dual and with the primal
param y_pr {1..m};
let {i in 1..m} y_pr[i] := if (gamma + sum {j in 1..n} w[j] * A_te[i,j]) <= 0 then -1 else 1;
param misclass_count default 0;
for {i in {1..m}} {
	if y_pr[i] != y_te[i] then
		let misclass_count := misclass_count + 1;
}
param accuracy = (m - misclass_count) / m;
display accuracy;

