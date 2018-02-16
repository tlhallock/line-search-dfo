//
// Created by thallock on 12/26/17.
//

#ifndef C_TRUSTREGION_H
#define C_TRUSTREGION_H

#include <armadillo>

#include "./Shifter.h"

class TrustRegionObj: public Shifter {
public:
    double radius;
    arma::vec center;

    TrustRegionObj(double radius, const arma::vec& center)
      : radius{radius}
      , center(center)
    {}

    arma::mat shift(const arma::mat& m) const;
    arma::mat unshift(const arma::mat& m) const;
    arma::vec shift(const arma::vec& m) const;
    arma::vec unshift(const arma::vec& m) const;
    bool contains(const  arma::vec& v) const;

};


#endif //C_TRUSTREGION_H
