#ifndef INCR_GMM_HPP
#define INCR_GMM_HPP


#include <iostream>
#include <map>

#include "classifier.hpp"
#include "component.hpp"
#include "gmm_estimator.hpp"

namespace  cmm {


class IncrementalGMM : public Classifier{
public:

    typedef std::map<int, std::vector<Component::Ptr>> model_t;

    IncrementalGMM(){
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }
    IncrementalGMM(int dimension, int nbr_class) :
        Classifier(dimension,nbr_class){

        for(int i = 0; i < nbr_class; i++)
            _model.emplace(i,std::vector<Component::Ptr>());

        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }

    IncrementalGMM(const model_t& model){
        _dimension = model.at(0)[0]->get_dimension();
        _nbr_class = model.size();
        for(const auto& comps : model){
            _model.emplace(comps.first,std::vector<Component::Ptr>());
            for(const auto& comp : comps.second)
                _model[comps.first].push_back(Component::Ptr(new Component(*(comp))));
        }
        srand(time(NULL));
//        _gen.seed(rand());
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }

    IncrementalGMM(const IncrementalGMM& igmm) :
        Classifier(igmm),_model(igmm._model),
    _last_index(igmm._last_index), _last_label(igmm._last_label),
    _alpha(igmm._alpha), _u(igmm._u), _beta(igmm._beta){}


    ~IncrementalGMM(){
        for(auto& comps: _model)
            for(auto& c : comps.second)
                c.reset();
    }

    std::vector<double> compute_estimation(const Eigen::VectorXd &X) const;
    model_t& model(){return _model;}
    const model_t& model() const {return _model;}

    void new_component(const Eigen::VectorXd& sample, int label);

    void add(const Eigen::VectorXd &sample, int lbl);
    void update();

    double confidence(const Eigen::VectorXd &sample) const;
    int next_sample(const std::vector<std::pair<Eigen::VectorXd, std::vector<double> > > &samples, Eigen::VectorXd &choice_dist_map);

    void update_factors();
    void _update_factors(int lbl);

    void set_alpha(double a){_alpha = a;}
    void set_u(double u){_u = u;}
    void set_beta(double b){_beta = b;}

private:

    bool _merge(const Component::Ptr& comp);
    bool _split(const Component::Ptr& comp);

    model_t _model;

    int _last_index = 0;
    int _last_label = 0;

//    bool _llhood_drive;

    double _alpha; /**<factor split parameter*/
    double _u; /**<mean split parameter*/
    double _beta; /**<covariance split parameter*/
};

} //cmm

#endif //INCR_GMM_HPP
