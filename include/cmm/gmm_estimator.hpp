#ifndef GMM_ESTIMATOR_HPP
#define GMM_ESTIMATOR_HPP

#ifndef NO_PARALLEL
#include <tbb/tbb.h>
#endif

#include <iostream>

#include <map>
#include <eigen3/Eigen/Core>

namespace cmm{

template <class gmm>
/**
 * @brief Helper class to estimate the prediction of CMM with parallel reduce algo of intel tbb
 */
class Estimator{
public:
    Estimator(const gmm* model, const Eigen::VectorXd& X, int lbl)
        : _X(X), _current_lbl(lbl),_sum(0){
        _model = new gmm(*model);
    }


#ifndef NO_PARALLEL
    Estimator(const Estimator& est, tbb::split) : _model(est._model), _X(est._X), _current_lbl(est._current_lbl){
        _sum = 0;
    }

    void operator()(const tbb::blocked_range<size_t>& r){
        double val;
        double sum = _sum;

        Eigen::VectorXd X = _X;

        for(size_t i=r.begin(); i != r.end(); ++i){
            val = _model->model()[_current_lbl][i]->get_factor()*
                    _model->model()[_current_lbl][i]->compute_multivariate_normal_dist(X);
            sum += val;

        }
        _sum = sum;
    }

    void join(const Estimator& est){
        _sum += est._sum;
    }
#endif

    double get_sum(){return _sum;}

private:
    gmm* _model;
    double _sum;
    int _current_lbl;
    Eigen::VectorXd _X;
};

template<class gmm>
/**
 * @brief estimation of the class of sample X from a classifier of type GMM (either CMM or incremental CMM)
 * @param the classifier
 * @param a sample
 * @return a vector containing the probability of membership to each class of sample X
 */
std::vector<double> estimation(const gmm* model, Eigen::VectorXd X){
    std::map<int,double> sum_map;
    for(int i = 0; i < model->get_nbr_class(); i++){
        sum_map[i] = 0;
    }
    std::vector<double> estimations;

#ifdef NO_PARALLEL
    for(int lbl = 0; lbl < model->get_nbr_class(); lbl++)
    {
        double val;
        for(const auto& comp : model->model().at(lbl))
        {
            val = comp->get_factor()*
                    comp->compute_multivariate_normal_dist(X);
            sum_map[lbl] += val;
        }
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0,model->get_nbr_class()),
                      [&](const tbb::blocked_range<size_t>& r){
        for(int lbl = r.begin(); lbl != r.end();lbl++){
            Estimator<gmm> estimator(model,X,lbl);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0,model->model().at(lbl).size()),estimator);
            sum_map[lbl] = estimator.get_sum();
        }
    });
#endif

    double sum_of_sums = 0;
    for(const auto& sum : sum_map)
        sum_of_sums += sum.second;
    for(int lbl = 0; lbl < model->get_nbr_class(); lbl++)
        estimations.push_back((1 + sum_map[lbl])/(model->get_nbr_class() + sum_of_sums));
    return estimations;
}

}//cmm
#endif //GMM_ESTIMATOR_HPP
