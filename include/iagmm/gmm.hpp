#ifndef GMM_HPP
#define GMM_HPP

#include <iostream>
#include <memory>
#include <math.h>
#include <vector>
#include <map>

#include "boost/random.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <tbb/tbb.h>

#include "component.hpp"
#include "classifier.hpp"

namespace iagmm{

class GMM : public Classifier {
public:

    typedef std::map<int, std::vector<Component::Ptr>> model_t;

    GMM(){}

    GMM(int dimension, int nbr_class) : Classifier(dimension,nbr_class){
        for(int i = 0; i < nbr_class; i++)
            _model.emplace(i,std::vector<Component::Ptr>());
    }

    GMM(const model_t& model){

        _dimension = model.at(0)[0]->get_dimension();
        _nbr_class = model.size();
        for(const auto& comps : model){
            _model.emplace(comps.first,std::vector<Component::Ptr>());
            for(const auto& comp : comps.second)
                _model[comps.first].push_back(Component::Ptr(new Component(*(comp))));
        }

    }

    GMM(const model_t& model, const Eigen::VectorXd& X) :
        _model(model),
        _X(X), _current_lbl(1){}

    GMM(GMM& gmm, tbb::split) :
        _model(gmm._model),
        _X(gmm._X), _current_lbl(gmm._current_lbl){
        _sum_map[_current_lbl] = 0;
    }

    ~GMM(){
        for(auto& comps: _model)
            for(auto& c : comps.second)
                c.reset();
    }

    void operator()(const tbb::blocked_range<size_t>& r);

    void join(const GMM& gmm){
        _sum_map[_current_lbl] += gmm._sum_map.at(_current_lbl);
    }

    double compute_estimation(const Eigen::VectorXd& sample, int lbl);
    model_t& model(){return _model;}

    double get_result(int lbl){
        double sum_of_sums;
        for(const auto& sum : _sum_map)
            sum_of_sums +=  sum.second;
        return _sum_map[lbl]/(sum_of_sums);
    }

    void append(const std::vector<Eigen::VectorXd> &samples,const std::vector<int>& lbl);
    int append(const Eigen::VectorXd &samples,const int& lbl);

    void update_model(int ind, int lbl);

    std::vector<int> find_closest_components(double& min_dist, int lbl);

    int find_closest(int i, double& min_dist, int lbl);

    void update_factors();
    double unit_factor();

    double model_score(const std::vector<Eigen::VectorXd>& samples, const std::vector<int> &label);

    std::vector<double> model_scores();

    double entropy(int i, int sign);

    int next_sample(const samples_t& samples, Eigen::VectorXd& choice_dist_map);

    /**
     * @brief k nearst neighbor
     * @param samples
     * @param output
     * @param k
     */
    void knn(const Eigen::VectorXd& center,TrainingData& output, int k);


    int number_of_samples(){
        int nbr_s = 0;
        for(const auto& components : _model)
            for(const auto& c : components.second)
                nbr_s += c->size();

        return nbr_s;
    }

    template <typename archive>
    void serialize(archive& arch, const unsigned int v){
        arch & _nbr_class;
        arch & _dimension;
        arch & _model;
    }

    std::string print_info();
    std::string to_string();

private:

    void _merge(int ind,int lbl);
    double _component_score(int i, int lbl);
    void _split(int ind ,int sign);
    void _new_component(const Eigen::VectorXd &samples, int label);


    model_t _model;

    Eigen::VectorXd _X;

    //variables for parallel computation
    std::map<int,double> _sum_map;
    int _current_lbl;

    boost::random::mt19937 _gen;

};
}

#endif //GMM_HPP
