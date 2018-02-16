//
// Created by thallock on 12/26/17.
//

#include "TrustRegion.h"

arma::mat TrustRegionObj::shift(const arma::mat& m) const {
    return (m - arma::ones(m.n_rows, 1) * center.t()) / radius;
}

arma::vec TrustRegionObj::unshift(const arma::vec& m) const {
    return m * radius + center;
}
arma::vec TrustRegionObj::shift(const arma::vec& m) const {
    return (m - center) / radius;
}

arma::mat TrustRegionObj::unshift(const arma::mat& m) const {
    return m * radius + arma::ones(m.n_rows, 1) * center.t();
}


bool TrustRegionObj::contains(const arma::vec& v) const {
    return arma::norm(v - center) <= radius;
}