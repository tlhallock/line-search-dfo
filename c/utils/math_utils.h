//
// Created by thallock on 12/16/17.
//

#ifndef C_MATH_UTILS_H
#define C_MATH_UTILS_H

#include "armadillo"


#include <cassert>
#include <cmath>


double nchoosek(int n, int k);
int factorial(int n);


double cumprod(const arma::vec& vec);

//template<class Object>
//Object elementwise_pow(const Object& base, const Object& p);



template<class Object>
Object elementwise_pow(const Object& base, const Object& p) {
    assert(base.n_elem == p.n_elem);
    Object result;
    result.copy_size(base);
    for (std::size_t i = 0; i < result.n_elem; ++i) {
        result[i] = std::pow(base[i], p[i]);
    }
    return result;
}

void copyToRow(arma::mat& m, int row, arma::vec y);

#endif //C_MATH_UTILS_H
