#include <iagmm/mcs.hpp>
#include <tbb/tbb.h>

using namespace iagmm;

double MCS::compute_estimation(const Eigen::VectorXd& sample, int lbl){
    std::vector<double> estimations;

    for(auto& classif : _classifiers)
        estimations.push_back(classif.compute_estimation(sample,lbl));

    return _comb_fct(_parameters,estimations);
}

void MCS::update(){
    estimations = Eigen::MatrixXd::Zero(_classifiers.size(),_samples.size());

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0,_classifiers.size(),0,_samples.size()),
                      [&](const tbb::blocked_range2d<size_t>& r ){
        for(size_t i = r.rows().begin(); i != r.rows().end(); i++){
            for(size_t j = r.cols().begin(); j != r.cols().end(); j++){
                estimations(i,j) = _classifiers[i].compute_estimation(_samples[j].second,_samples[j].first);
    });

    _opt_fct(_parameters,_classifiers,_samples);
}
