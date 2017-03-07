#include <iagmm/nnmap.hpp>

using namespace iagmm;


double NNMap::compute_estimation(const Eigen::VectorXd &sample, int label){

    double estimation = 0.5;

    for(const auto& s : _samples.get()){
        double dist = (s.second - sample).squaredNorm();

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
