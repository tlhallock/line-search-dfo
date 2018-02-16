//
// Created by thallock on 12/23/17.
//

#ifndef C_BOUNDS_H
#define C_BOUNDS_H

#include <armadillo>
#include <cmath>

class Bounds {
public:
    arma::vec upper;
    arma::vec lower;

    Bounds(int n)
      : upper(n, arma::fill::none)
      , lower(n, arma::fill::none)
    {
        upper.fill(NAN);
        lower.fill(NAN);
    }


    void should_include(const arma::vec& point) {
        if (upper.n_elem != point.n_elem || lower.n_elem != point.n_elem) {
            throw 1;
        }
        for (int i=0;i<point.n_elem;i++) {
            if (std::isnan(upper(i)) || upper(i) < point(i)) {
                upper(i) = point(i);
            }
            if (std::isnan(lower(i)) || lower(i) > point(i)) {
                lower(i) = point(i);
            }
        }
    }
};


#endif //C_BOUNDS_H
