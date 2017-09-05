#include <iostream>
#include <iagmm/mcs.hpp>
#include <tbb/tbb.h>

using namespace iagmm;

const std::map<std::string,comb_fct_t> combinatorial::fct_map = combinatorial::create_map();

double MCS::compute_estimation(const Eigen::VectorXd& sample, int lbl){
    std::vector<double> estimations;

    for(auto& classif : _classifiers)
        estimations.push_back(classif->compute_estimation(sample,lbl));

    return _comb_fct(_parameters,estimations);
}

void MCS::add(const Eigen::VectorXd &sample, int lbl){
    _samples.add(lbl,sample);
    for(auto& c : _classifiers)
        c->add(sample,lbl);

}

void MCS::update(){

    for(auto& c : _classifiers)
        c->update();

    int c_size = _classifiers.size();
    int s_size = _samples.size();

    _estimations = Eigen::MatrixXd::Zero(s_size,c_size);
    Eigen::VectorXd lbl(s_size);


    tbb::parallel_for(tbb::blocked_range2d<size_t>(0,c_size,0,s_size),
                      [&](const tbb::blocked_range2d<size_t>& r ){
        for(size_t i = r.rows().begin(); i != r.rows().end(); i++)
            for(size_t j = r.cols().begin(); j != r.cols().end(); j++)
                _estimations(j,i) = _classifiers[i]->compute_estimation(_samples[j].second,_samples[j].first);
    });


//    for(int i = 0; i < c_size; i++){
//        lbl(i) = _samples[i].first;
//    }

//    _conjugate_grad.compute(_estimations);
//    _parameters = _conjugate_grad.solve(lbl);
//    std::cout << _parameters << std::endl;

//    std::cout <<  _conjugate_grad.iterations() << " error : " << _conjugate_grad.error() << std::endl;
}
