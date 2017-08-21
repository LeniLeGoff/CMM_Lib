#ifndef MCS_HPP
#define MCS_HPP

#include "mcs_fct.hpp"
#include "classifier.hpp"
#include <vector>

namespace iagmm
{
class MCS : public Classifier {

public:
    MCS(comb_fct_t& cmb, opt_fct_t& opt):
        _comb_fct(cmb),_opt_fct(opt){
    }
    MCS(std::vector<Classifier::Ptr>& class_list, comb_fct_t& cmb, opt_fct_t& opt):
        _classifiers(class_list), _comb_fct(cmb),_opt_fct(opt){    }
    MCS(const MCS& mcs) :
        Classifier(mcs), _parameters(mcs._parameters),
        _classifiers(mcs._classifiers), _comb_fct(mcs._comb_fct),
        _opt_fct(mcs._opt_fct){}


    double compute_estimation(const Eigen::VectorXd &sample, int lbl);

    void update();

private:
    Eigen::VectorXd _parameters;
    std::vector<Classifier::Ptr> _classifiers;
    Eigen::MatrixXd _estimations;
    comb_fct_t _comb_fct;
    opt_fct_t _opt_fct;

};
}

#endif // MCS_HPP
