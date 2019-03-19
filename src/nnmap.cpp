#include <cmm/nnmap.hpp>

using namespace cmm;


std::vector<double> NNMap::compute_estimation(const Eigen::VectorXd &sample) const {

    double estimation = default_estimation;

    for(const auto& s : _samples.get()){
        double dist = _distance(s.second,sample);

        if(dist < distance_threshold){
            if(s.first == 1)
                estimation += increment_factor*(1-dist);
            else estimation -= increment_factor*(1-dist);

            if(estimation < 0)
                estimation = 0;
            if(estimation > 1)
                estimation = 1.;
        }
    }

    return {1-estimation,estimation};

}
