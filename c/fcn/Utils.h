//
// Created by thallock on 12/19/17.
//

#ifndef C_UTILS_H
#define C_UTILS_H

#include <armadillo>

class Function {
public:
    virtual double evaluate(const arma::vec& x) const // = 0;
    {
        throw "not implemented";
    }
    virtual arma::vec gradient(const arma::vec& x) const // = 0;
    {
        throw "not implemented";
    }
    virtual arma::mat hessian(const arma::vec& x) const //  = 0;
    {
        throw "not implemented";
    }
};

class Quadratic: public Function {
public:
    arma::mat Q;
    arma::vec b;
    double c;

    Quadratic(arma::mat Q, arma::vec b, double c)
        : Q(Q)
        , b(b)
        , c(c)
    {}

    double evaluate(const arma::vec& x) const {
        return  0.5 * arma::as_scalar(x.t() * Q * x + b.t() * x) + c;
    }
    arma::vec gradient(const arma::vec& x) const {
        return Q * x + b;
    }
    arma::mat hessian(const arma::vec& x) const {
        return Q;
    }
};

class Plane: public Function {
public:
    arma::vec n;
    double b;

    Plane(arma::vec n, double b)
            : n(n)
            , b(b)
    {}

    double evaluate(const arma::vec& x) const {
        return arma::dot(n, x) - b;
    }
    arma::vec gradient(const arma::vec& x) const {
        return n;
    }
    arma::mat hessian(const arma::vec& x) const {
        return arma::zeros(n.n_elem, n.n_elem);
    }
};

class BasisFunction: public Function {
public:
    arma::vec coefficients;
    std::function<arma::vec(const arma::vec&)> toRow;
    std::function<arma::mat(const arma::vec&)> toGradients;

    BasisFunction(
        const arma::vec& coeff,
        std::function<arma::vec(const arma::vec&)> toRow,
        std::function<arma::mat(const arma::vec&)> toGradients
    )
      : coefficients(coeff)
      , toRow(toRow)
      , toGradients(toGradients)
    {}

    double evaluate(const arma::vec& x) const {
        return arma::dot(toRow(x), coefficients);
    }
    arma::vec gradient(const arma::vec& x) const {
        return toGradients(x) * coefficients;
    }
    arma::mat hessian(const arma::vec& x) const {
        throw 1;
    }
};


class TrustRegion: public Function {
public:
    double radius;
    arma::vec center;

    TrustRegion(
      double radius = 1,
      const arma::vec center = {0, 0}
    )
      : radius{radius}
      , center(center)
    {}

    double evaluate(const arma::vec& x) const {
        return arma::dot(x - center, x - center) - radius * radius;
    }
    arma::vec gradient(const arma::vec& x) const {
        return 2 * (x - center);
    }
    arma::mat hessian(const arma::vec& x) const {
        return 2 * arma::eye(x.n_elem, x.n_elem);
    }
};

class Negation: public Function {
public:
    Function *delegate;

    Negation(
      Function *delegate
    )
      : delegate{delegate}
    {}

    double evaluate(const arma::vec& x) const {
        return -delegate->evaluate(x);
    }
    arma::vec gradient(const arma::vec& x) const {
        return -delegate->gradient(x);
    }
    arma::mat hessian(const arma::vec& x) const {
        return -delegate->hessian(x);
    }
};

#endif //C_UTILS_H
