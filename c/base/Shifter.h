//
// Created by thallock on 12/28/17.
//

#ifndef C_SHIFTER_H
#define C_SHIFTER_H

#include <armadillo>

class Shifter {
public:
    virtual arma::vec shift(const arma::vec&) const = 0;
    virtual arma::vec unshift(const arma::vec&) const = 0;
    virtual arma::mat shift(const arma::mat&) const = 0;
    virtual arma::mat unshift(const arma::mat&) const = 0;
};


#endif //C_SHIFTER_H
