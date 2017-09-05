#ifndef MCS_HPP
#define MCS_HPP

#include <iagmm/mcs_fct.hpp>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include "boost/random.hpp"


namespace iagmm
{
class MCS : public Classifier {

public:
    MCS(){}
    MCS(std::map<std::string,Classifier::Ptr>& class_list,const comb_fct_t& cmb):
        _classifiers(class_list), _comb_fct(cmb){
        _dimension = _classifiers[0]->get_dimension();
        _nbr_class = _classifiers[0]->get_nbr_class();
        _parameters = Eigen::VectorXd::Constant(class_list.size(),1.);
    }
    MCS(const MCS& mcs) :
        Classifier(mcs), _parameters(mcs._parameters),
        _classifiers(mcs._classifiers), _comb_fct(mcs._comb_fct){
    }

    double compute_estimation(const std::map<std::string,Eigen::VectorXd> &sample, int lbl);

    void add(const Eigen::VectorXd &sample, int lbl){}
    void add(const std::map<std::string,Eigen::VectorXd> &sample, int lbl);

    void update();

    int next_sample(std::vector<std::pair<Eigen::VectorXd,double>> samples, Eigen::VectorXd& choice_dist_map);

    std::map<std::string,Classifier::Ptr>& access_classifiers(){return _classifiers;}

private:
    Eigen::VectorXd _parameters;
    std::map<std::string,Classifier::Ptr> _classifiers;
    Eigen::MatrixXd _estimations;
    comb_fct_t _comb_fct;
    boost::random::mt19937 _gen;

};
}

#endif // MCS_HPP
