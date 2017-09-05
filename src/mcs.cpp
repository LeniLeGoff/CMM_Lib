#include <iostream>
#include <iagmm/mcs.hpp>
#include <tbb/tbb.h>

using namespace iagmm;

const std::map<std::string,comb_fct_t> combinatorial::fct_map = combinatorial::create_map();

double MCS::compute_estimation(const std::map<std::string, Eigen::VectorXd> &sample, int lbl){
    std::vector<double> estimations;

    int i = 0;
    for(auto& classif : _classifiers){
        estimations.push_back(classif.second->compute_estimation(sample.at(classif.first),lbl));
        _parameters[i] = classif.second->confidence(sample.at(classif.first));
        i++;
    }

    return _comb_fct(_parameters,estimations);
}

void MCS::add(const std::map<std::string,Eigen::VectorXd> &sample, int lbl){
    for(auto& c : _classifiers)
        c.second->add(sample.at(c.first),lbl);
}

void MCS::update(){
    for(auto& c : _classifiers)
        c.second->update();
}

int MCS::next_sample(std::vector<std::pair<Eigen::VectorXd, double> > samples, Eigen::VectorXd& choice_dist_map){
    std::vector<int> indexes;
    std::vector<Eigen::VectorXd> cdms;
    for(auto& c: _classifiers){
        indexes.push_back(c.second->next_sample(samples,choice_dist_map));
        cdms.push_back(choice_dist_map);
    }

    boost::random::uniform_int_distribution<> distrib(0,indexes.size()-1);
    int choice = distrib(_gen);
    choice_dist_map = cdms[choice];

    return indexes[choice];
}
