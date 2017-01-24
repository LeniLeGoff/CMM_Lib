#ifndef GMM_HPP
#define GMM_HPP

#include <iostream>
#include <memory>
#include <math.h>
#include <vector>

#include "boost/random.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <tbb/tbb.h>

#include "component.hpp"

namespace iagmm{

class GMM{
public:

    typedef std::vector<Component::Ptr> model_t;

    GMM() : _pos_sum(0), _neg_sum(0){}

    GMM(const model_t& pos_components,
        const model_t& neg_components) :
        _pos_sum(0), _neg_sum(0)
    {

        for(const auto& elt : pos_components)
            _pos_components.push_back(Component::Ptr(new Component(*(elt))));
        for(const auto& elt : neg_components)
            _neg_components.push_back(Component::Ptr(new Component(*(elt))));

    }

    GMM(const model_t& pos_components,
        const model_t& neg_components , const Eigen::VectorXd& X) :
        _pos_components(pos_components), _neg_components(neg_components),
        _X(X), _pos_sum(0), _neg_sum(0), _sign(1){}

    GMM(GMM& gmm, tbb::split) :
        _pos_components(gmm._pos_components),
        _neg_components(gmm._neg_components),
        _X(gmm._X) , _pos_sum(0), _neg_sum(0), _sign(gmm._sign){}

    ~GMM(){
        for(auto& elt: _pos_components)
            elt.reset();

        for(auto& elt: _neg_components)
            elt.reset();
    }

    void operator()(const tbb::blocked_range<size_t>& r);

    void join(const GMM& gmm){
        if(_sign > 0)
            _pos_sum+=gmm._pos_sum;
        else _neg_sum+=gmm._neg_sum;
    }

    double compute_GMM(Eigen::VectorXd X);
    model_t& get_pos_components(){return _pos_components;}
    model_t& get_neg_components(){return _neg_components;}

    double get_result(){return _pos_sum/(_pos_sum+_neg_sum);}

    void update_model(const std::vector<Eigen::VectorXd>& samples, const std::vector<double> &label);

    std::vector<int> find_closest_components(double& min_dist, double sign);

    int find_closest(int i, double& min_dist, double sign);

    void update_factors();
    double unit_factor();

    double model_score(const std::vector<Eigen::VectorXd>& samples, const std::vector<double> &label);

    std::vector<double> model_scores();


    double entropy(int i, int sign);

    Eigen::VectorXd next_sample(Eigen::MatrixXd &means_entropy_map);

    /**
     * @brief k nearst neighbor
     * @param samples
     * @param output
     * @param k
     */
    void knn(const Eigen::VectorXd& center, const std::vector<Eigen::VectorXd>& samples
             , const std::vector<double> &label, std::vector<Eigen::VectorXd>& output,
             std::vector<double> &label_output, int k);

    int number_of_samples(){
        int res = 0;
        for(const auto& c : _pos_components)
            res += c->size();
        for(const auto& c : _neg_components)
            res += c->size();

        return res;
    }


private:

    void _merge(int sign);
    double _component_score(int i, int sign);
    void _split(int sign, const std::vector<Eigen::VectorXd>& samples, const std::vector<double> &label);
    void _new_component(const std::vector<Eigen::VectorXd> &samples, double label);

    model_t _pos_components;
    model_t _neg_components;

    Eigen::VectorXd _X;
    double _pos_sum;
    double _neg_sum;
    int _sign;

    boost::random::mt19937 _gen;

};
}

#endif //GMM_HPP
