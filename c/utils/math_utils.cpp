//
// Created by thallock on 12/16/17.
//


#include <cassert>
#include <cmath>


#include "math_utils.h"

// not that stable
double nchoosek(int n, int k) {
    double result = 1;
    for (int i=n-k+1; i<=n; i++) {
        result *= i;
    }
    for (int i=1; i<=k; i++) {
        result /= i;
    }
    return result;
}

int factorial(int n) {
    int result = 1;
    for (int i=2; i<=n; i++) {
        result *= i;
    }
    return result;
}



// arma::cumprod  ????
//template<class Object>
double cumprod(const arma::vec& vec) {
    double result = 1;
    for (std::size_t i=0; i<vec.n_elem; i++) {
        result *= vec[i];
    }
    return result;
}

void copyToRow(arma::mat& m, int row, arma::vec y) {
    for (int i=0;i<y.n_elem;i++) {
        m[row, i] = y[i];
    }
}