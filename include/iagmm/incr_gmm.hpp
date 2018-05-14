#include <iostream>
#include <map>

#include "classifier.hpp"
#include "component.hpp"

namespace  iagmm {


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
        Classifier(igmm),_model(igmm._model){}

    std::vector<double> compute_estimation(const Eigen::VectorXd &sample);
    model_t& model(){return _model;}

    void new_component(const Eigen::VectorXd& sample, int label);

    void add(const Eigen::VectorXd &sample, int lbl);
    void update();

    void update_factors();
    void _update_factors(int lbl);

private:

    bool _merge(const Component::Ptr& comp);
    bool _split(const Component::Ptr& comp);

    model_t _model;

    int _last_index;
    int _last_label;

//    bool _llhood_drive;

    double _alpha; /**<factor split parameter*/
    double _u; /**<mean split parameter*/
    double _beta; /**<covariance split parameter*/
};

} //iagmm
