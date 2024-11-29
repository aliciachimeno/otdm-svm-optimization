reset;

# solve primal
print "SVM PRIMAL:";

model svm-primal.mod;
data "./data/generated/ampl_format/data1000_converted.dat"
#data "./data/real/ampl_format/real_dataset.dat"
option solver cplex; 

problem SVM_PRIMAL: w, gamma, s, svm_primal, c1, c2;
solve SVM_PRIMAL;
display w, gamma, s;


param y_pr {1..m};
let {i in 1..m} y_pr[i] := if (gamma + sum {j in 1..n} w[j] * A_te[i,j]) <= 0 then -1 else 1;
param misclass_count default 0;
for {i in {1..m}} {
	if y_pr[i] != y_te[i] then
		let misclass_count := misclass_count + 1;
}
param accuracy = (m - misclass_count) / m;
display accuracy;

