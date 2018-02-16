//
// Created by thallock on 12/16/17.
//

#include "PolynomialBasis.h"
#include "armadillo"

#include "../utils/math_utils.h"

static int find_all_powers(arma::Col<int>& current, int index, int balls, int idxToDest, arma::Mat<int>& destination) {
    if (balls < 0 || index >= current.n_elem) {
        return idxToDest;
    }
    if (index == current.n_elem - 1) {
        current[index] = balls;
        destination.col(idxToDest) = current;
        return idxToDest + 1;
    }
    for (int rem = 0; rem <= balls; rem++) {
        current[index] = rem;
        idxToDest = find_all_powers(current, index + 1, balls - rem, idxToDest, destination);
    }
    return idxToDest;
}

static int get_basis_dimension(int n, int degree) {
    int count = 0;
    for (int i = 0; i <= degree; i++) {
        count += nchoosek(i+n-1, n-1);
    }
    return count;
}

PolynomialBasis::PolynomialBasis(int n, int degree)
    : n{n}
    , degree{degree}
    , basis_dimension{get_basis_dimension(n, degree)}
    , powers(n, basis_dimension)
    , coeff(basis_dimension) {
    int index = 0;
    arma::Col<int> current(n, arma::fill::zeros);
    for (int i=0;i<=degree;i++) {
        int num = nchoosek(i+n-1, n-1);
        double c = 1.0 / factorial(i);
        for (int j=index;j<index+num;j++) {
            coeff[j] = c;
        }
        index = find_all_powers(current,  0, i, index, powers);
    }
}


arma::vec PolynomialBasis::toRow(const arma::vec& x) const {
    if (x.n_elem != n) {
        throw 1;
    }
    arma::vec result(powers.n_cols);

    for (int i=0; i<powers.n_cols; i++) {
        arma::vec v = elementwise_pow(x, arma::conv_to<arma::vec>::from(powers.col(i)));
        result[i] = cumprod(v) * coeff[i];
    }

    return result;
}

arma::mat PolynomialBasis::toRow(const arma::mat& x) const {
    if (x.n_cols != n) {
        throw 1;
    }
    arma::mat result(x.n_rows, powers.n_cols);

    for (int i=0; i<x.n_rows; i++) {
        arma::vec v = toRow(arma::conv_to<arma::vec>::from(x.row(i)));
        result.row(i) = arma::conv_to<arma::Row<double> >::from(v);
    }

    return result;
}

/*
 * Should return the jacobian...
 */
arma::mat PolynomialBasis::gradients(const arma::vec& x) const {
    if (x.n_elem != n || x.n_cols != 1) {
        throw 1;
    }
    arma::mat result(x.n_rows, powers.n_cols);

    for (int var = 0; var < x.n_rows; var++) {
        for (int i=0; i<powers.n_cols; i++) {
            arma::vec pows = arma::conv_to<arma::vec>::from(powers.col(i));
            if (pows[var] == 0) {
                result(var, i) = 0;
                continue;
            }
            double c = pows[var]--;
            arma::vec v = elementwise_pow(x, pows);
            result(var, i) = cumprod(v) * coeff[i] * c;
        }
    }

    return result;
}


BasisFunction *PolynomialBasis::createPolynomial(const arma::vec& coefficients) const {
    return new BasisFunction(
      coefficients,
      [this](const arma::vec& x) { return toRow(x); },
      [this](const arma::vec& x) { return gradients(x); }
    );
}


std::ostream& operator<<(std::ostream& os, const PolynomialBasis& basis) {
    os << "n:" << basis.n << '\n';
    os << "degree:" << basis.degree << '\n';
    os << "basis_dimenion:" << basis.basis_dimension << '\n';
    os << "powers:\n" << basis.powers << '\n';
    os << "coeff:\n" << basis.coeff << '\n';
    return os;
}