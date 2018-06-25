#ifndef GMM_ESTIMATOR_HPP
#define GMM_ESTIMATOR_HPP

#ifndef NO_PARALLEL
#include <tbb/tbb.h>
#endif

#include <iostream>

#include <map>
#include <eigen3/Eigen/Core>

namespace iagmm{

template <class gmm>
class Estimator{
public:
    Estimator(gmm* model, const Eigen::VectorXd& X)
        : _model(model), _X(X), _current_lbl(0){


        for(int i = 0; i < _model->model().size(); i++)
            _sum_map.emplace(i,0.);
    }


#ifndef NO_PARALLEL
    Estimator(const Estimator& est, tbb::split) : _model(est._model), _X(est._X), _current_lbl(est._current_lbl){
        _sum_map[_current_lbl] = 0;
    }

    void operator()(const tbb::blocked_range<size_t>& r){
        double val;
        double sum = _sum_map[_current_lbl];

        Eigen::VectorXd X = _X;

        for(size_t i=r.begin(); i != r.end(); ++i){
            val = _model->model()[_current_lbl][i]->get_factor()*
                    _model->model()[_current_lbl][i]->compute_multivariate_normal_dist(X);
            sum += val;

        }
        _sum_map[_current_lbl] = sum;
    }

    void join(const Estimator& est){
        _sum_map[_current_lbl] += est._sum_map.at(_current_lbl);
    }
#endif

    std::vector<double> estimation(){
#ifdef NO_PARALLEL
        for(_current_lbl = 0; _current_lbl < _model->get_nbr_class(); _current_lbl++)
        {
            double val;
            for(const auto& model : _model->model()[_current_lbl])
            {
                val = model->get_factor()*
                        model->compute_multivariate_normal_dist(_X);
                _sum_map[_current_lbl] += val;
            }
        }
#else
        tbb::parallel_for(tbb::blocked_range<size_t>(0,_model->get_nbr_class()),
                          [&](const tbb::blocked_range<size_t>& r){
            for(int lbl = r.begin(); lbl != r.end();lbl++){
                double val;
                for(const auto& model : _model->model()[lbl])
                {
                    val = model->get_factor()*
                            model->compute_multivariate_normal_dist(_X);
                    _sum_map[lbl] += val;
                }
            }

        });

//        for(_current_lbl = 0; _current_lbl < _model->get_nbr_class(); _current_lbl++)
//            tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_model->model()[_current_lbl].size()),*this);
#endif

        double sum_of_sums = 0;
        for(const auto& sum : _sum_map)
            sum_of_sums += sum.second;
        std::vector<double> estimations;
        for(int lbl = 0; lbl < _model->get_nbr_class(); lbl++)
            estimations.push_back((1 + _sum_map[lbl])/(_model->get_nbr_class() + sum_of_sums));
        return estimations;
    }

private:
    gmm* _model;
    std::map<int,double> _sum_map;
    int _current_lbl;
    Eigen::VectorXd _X;
};

}

#endif //GMM_ESTIMATOR_HPP
