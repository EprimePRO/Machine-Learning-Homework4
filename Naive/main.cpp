#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <armadillo>
#include <math.h>

using namespace arma;
using namespace std;

string fname = "titanic_project.csv";

class Passenger {
    public:
    int survived;
    int pclass;
    double age;
    int sex;

    string toString(){
        return to_string(pclass) + " " + to_string(survived) + " " + to_string(sex) + " " + to_string(age);
    }
};

vector<Passenger> readCSVData() {

    ifstream dataFile(fname);

    vector<Passenger> data;

    string line;
    getline(dataFile, line);
    while (getline(dataFile, line)) {
        Passenger currPassenger;
        stringstream currLine(line);
        string token;
        for (int i=0;i<5;i++) {
            getline(currLine, token, ',');
            switch(i){
                case 0: break;
                case 1:
                    currPassenger.pclass = stoi(token);
                    break;
                case 2:
                    currPassenger.survived = stoi(token);
                    break;
                case 3:
                    currPassenger.sex = stoi(token);
                    break;
                case 4:
                    currPassenger.age = stod(token);
                    break;
                default: break;
            }
        }
        data.push_back(currPassenger);
    }


    return data;

}

mat get_count_survived(vector<Passenger>& pass){
    mat count(1,2); count.zeros();

    for(auto p: pass){
        if(p.survived==0){
            count(0,0)++;
        } else {
            count(0,1)++;
        }
    }

    return count;
}

mat get_survived_prior(vector<Passenger>& pass){
    mat priors(1,2); priors.zeros();
    mat count = get_count_survived(pass);
    priors(0,0) = count(0,0)/pass.size();
    priors(0,1) = count(0,1)/pass.size();
    
    return priors;
}

mat get_likelihood_pclass(vector<Passenger>& pass, mat count){
    mat likelihood(2,3); likelihood.zeros();
    for(int sv =0; sv<2; sv++){
        for(int pc=0; pc<3; pc++){
            int numSurvived = 0;
            for(auto p: pass){
                if(p.survived==sv and p.pclass==(pc+1)){
                    numSurvived++;
                }
            }
            likelihood(sv, pc)  = numSurvived/count(0, sv);
        }
    }
    return likelihood;
}

mat get_likelihood_sex(vector<Passenger>& pass, mat count){
    mat likelihood(2,2); likelihood.zeros();
    for(int sv =0; sv<2; sv++){
        for(int sx=0; sx<2; sx++){
            int numSurvived = 0;
            for(auto p: pass){
                if(p.survived==sv and p.sex==sx){
                    numSurvived++;
                }
            }
            likelihood(sv, sx)  = numSurvived/count(0, sv);
        }
    }
    return likelihood;
}

//this fuction will return mean and variance matrix for age. top row being mean
// [mean survived==0, mean survived==1]
// [var  sruvived==0, var  survived==1]
mat get_mean_var_age(vector<Passenger>& pass, mat count){
    mat mean_var(2,2); mean_var.zeros();
    for(int sv=0; sv<2; sv++){
        //calculate mean first
        double age_sum = 0;
        for(auto p: pass){
            if(p.survived==sv){
                age_sum+=p.age;
            }
        }
        mean_var(0,sv) = age_sum/count(0,sv);

        //calculate variance
        long double sum_of_squares = 0;
        for(auto p: pass){
            if(p.survived==sv){
                sum_of_squares+=pow(p.age-mean_var(0,sv), 2.0);
            }
        }
        mean_var(1,sv) = sum_of_squares/(count(0,sv)-1);

    }
    return mean_var;
}

const double pi = 3.14159265358979323846;
//function to calculate age likelihood
//run like this: calc_age_lh(6, 25.9, 138)
double calc_age_lh(double age, double mean, double var){
    return 1/sqrt(2*pi*var)*exp(-(pow(age-mean, 2)/(2*var)));
}

//function used to calculate raw probabilities
mat calc_raw_prob(int pclass, int sex, double age, mat& lh_pclass, mat&lh_sex, mat&age_mean_var, mat&apriori){
    mat raw_prob(1,2); raw_prob.zeros();

    pclass-=1;//for indexing purposes
    double num_s = lh_pclass(1, pclass)*lh_sex(1, sex)*apriori(0, 1)*calc_age_lh(age, age_mean_var(0,1), age_mean_var(1,1));
    double num_p = lh_pclass(0, pclass)*lh_sex(0, sex)*apriori(0, 0)*calc_age_lh(age, age_mean_var(0,0), age_mean_var(1,0));
    double denominator =    num_s + num_p;
    raw_prob(0,1) = num_s / denominator;
    raw_prob(0,0) = num_p / denominator;

    return raw_prob;
}

int main(){
    
    vector<Passenger> passengers = readCSVData();
    vector<Passenger> test;
    vector<Passenger> train;
    //get the first 900 instances as train subjects
    for(int i=0; i<900; i++){
        train.push_back(passengers[i]);
    }
    //train
    for(int i=900; i<passengers.size(); i++){
        test.push_back(passengers[i]);
    }

    //get apriori and counts
    mat apriori = get_survived_prior(train);
    mat count = get_count_survived(train);
    //cout << "apriori:" << endl << apriori << endl << "count:" << endl << count << endl;

    //get likelihood for class and sex
    mat lh_pclass = get_likelihood_pclass(train, count);
    mat lh_sex = get_likelihood_sex(train, count);
    //cout << "lh_pclass:" << endl << lh_pclass << endl << "lh_sex:" << endl << lh_sex << endl;
    
    //get mean and variance for age
    mat age_mean_var = get_mean_var_age(train, count);
    //cout << "[mean]:" << endl << "[var ]:" << endl << age_mean_var << endl;
    
    //calculate probabilities for the data
    for(auto p:test){
        mat raw = calc_raw_prob(p.pclass, p.sex, p.age, lh_pclass, lh_sex, age_mean_var, apriori);
        //double raw = calc_age_lh(p.age, age_mean_var(0,0), age_mean_var(1,0));
        //cout << raw << endl;
    }


    return 0;
}