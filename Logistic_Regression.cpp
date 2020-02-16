#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <bits/stdc++.h>
#include <algorithm>
#include <armadillo>

using namespace arma;
using namespace std;

class plasma
{
private:
    /* data */
    int index;
    float fibrinogen;
    int globulin;
    int ESR;

public:
    plasma(){ index = 0; fibrinogen = 0.0; globulin = 0; ESR = 0;};

    void setData(vector<string> info){
        index = stoi(info.at(0));
        fibrinogen = stof(info.at(1));
        globulin = stoi(info.at(2));
        ESR = stoi(info.at(3));
        info.clear();
    };
    void to_String(){
        cout << index << " " << fibrinogen << " " << globulin << " " << ESR << " " << endl;
    }

    double getFib(){
        return fibrinogen;
    }

    int getESR(){
        return ESR;
    }

    ~plasma();
};

plasma::~plasma()
{
}

//takes in an input matrix and outputs a vector range [0,1]


mat sigmoid(mat z)
{
    mat result_mat(z.n_rows, 1);
    for(int i=0; i < result_mat.n_rows; i++){
        result_mat(i, 0) = 1.0/(1.0+exp(-z(i,0)));
    }
    return result_mat;
}




int main(){
    string index = "",line = "";
    ifstream input("Plasma.csv");

    //Check that the file exists
    if(input.is_open()){
        cout << "File can be read" << endl << endl;
        input.close();
    }

    //Read the input into the two variables
    input.open("Plasma.csv");
    plasma obj;

    vector <string> info;
    vector<plasma> matrix;
    //Parse the information into the correct vectors
    while(getline(input, line)){

        stringstream ss(line);
        string parse = "";
        //Tokenize
        while(getline(ss, parse, ',')){
            std::replace(parse.begin(), parse.end(), '\"', ' ');
            //cout << parse << endl;
            if(parse.compare(" ESR < 20 ") == 0){
                parse = "0";
            }
            else if (parse.compare(" ESR > 20 ") == 0){
                parse = "1";
            }
            info.push_back(parse);
        }
        //initialize object
        obj.setData(info);
        matrix.push_back(obj);
        info.clear();



    }

    input.close();

    //initialize the data, labels and weights matrix
    mat weights_mat(2, 1, fill::ones);
    mat data_mat(matrix.size(), 2, fill::ones);
    mat labels_mat(matrix.size(), 1, fill::ones);

    for (int i = 0; i < data_mat.n_rows; i++){
        data_mat(i, 1) = matrix[i].getFib();
    }

    for (int i=0; i< labels_mat.n_rows; i++){
        labels_mat(i,0) = matrix[i].getESR();
    }

    double learning_rate = 0.001;
    mat probability;
    mat error;
    for(int i=0; i <=500000;i++){
        probability = sigmoid(data_mat*weights_mat);
        error = labels_mat - probability;
        weights_mat = weights_mat + learning_rate * (data_mat.t() * error);
    }
    
    cout << weights_mat << endl;

    return 0;
}