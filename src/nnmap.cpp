#include <iagmm/nnmap.hpp>

using namespace iagmm;


double NNMap::compute_estimation(const Eigen::VectorXd &sample, int label){

    double estimation = default_estimation;

    for(const auto& s : _samples.get()){
        double dist = _distance(s.second,sample);

        if(dist < distance_threshold){
            if(s.first == label)
                estimation += increment_factor*(1-dist);
            else estimation -= increment_factor*(1-dist);

            if(estimation < 0)
                estimation = 0;
            if(estimation > 1)
                estimation = 1.;
        }
    }

    return estimation;

}

void NNMap::fit(const Eigen::VectorXd& sample, const int& label){
    add(sample, label);
}
