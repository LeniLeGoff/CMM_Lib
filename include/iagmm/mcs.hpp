#ifndef MCS_HPP
#define MCS_HPP

#include <iagmm/mcs_fct.hpp>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include "boost/random.hpp"
#include <tbb/tbb.h>


namespace iagmm
{
class MCS{

public:


    MCS(){
        srand(time(NULL));
        _gen.seed(rand());
    }
    MCS(std::map<std::string,Classifier::Ptr>& class_list,const comb_fct_t& cmb,const param_fct_t param_fct):
        _classifiers(class_list), _comb_fct(cmb), _param_fct(param_fct){
        _dimension = _classifiers.begin()->second->get_dimension();
        _nbr_class = _classifiers.begin()->second->get_nbr_class();
        _parameters = Eigen::VectorXd::Constant(class_list.size(),1.);
        srand(time(NULL));
        _gen.seed(rand());
    }
    MCS(const MCS& mcs) : _dimension(mcs._dimension), _nbr_class(mcs._nbr_class), _parameters(mcs._parameters),
        _classifiers(mcs._classifiers), _comb_fct(mcs._comb_fct), _param_fct(mcs._param_fct),_estimations(mcs._estimations){
        srand(time(NULL));
        _gen.seed(rand());
    }



    std::vector<double> compute_estimation(const std::map<std::string,Eigen::VectorXd> &sample);

    void add(const std::map<std::string,Eigen::VectorXd> &sample, int lbl);

    void update();

    void update_parameters(int label, double thres = 0.5, double rate = 0.1);

    int next_sample(const std::map<std::string, std::vector<std::pair<Eigen::VectorXd, std::vector<double>> > > &samples, Eigen::VectorXd& choice_dist_map);
    double confidence(const Eigen::VectorXd& sample){}

    std::map<std::string,Classifier::Ptr>& access_classifiers(){return _classifiers;}

    void set_samples(std::string mod, TrainingData& data);
    int get_nb_samples(){return _classifiers.begin()->second->get_samples().size();}
    const TrainingData& get_samples(){return _classifiers.begin()->second->get_samples();}
    int get_nbr_class(){return _nbr_class;}

private:
    int _dimension;
    int _nbr_class;
    Eigen::VectorXd _parameters;
    std::map<std::string,Classifier::Ptr> _classifiers;
    std::vector<double> _estimations;
    comb_fct_t _comb_fct;
    param_fct_t _param_fct;
    boost::random::mt19937 _gen;

};
}

#endif // MCS_HPP
