#include <iostream>
#include <yaml-cpp/yaml.h>
#include <iagmm/data.hpp>
#include <boost/algorithm/string.hpp>

int main(int argc, char** argv){
    if(argc < 2){
        std::cout << "Usage : path to a yml file" << std::endl;
        return 1;
    }

    iagmm::TrainingData data;
    int dimension, nbr_class;
    data.load_yml(argv[1],dimension,nbr_class);
    std::vector<std::string> stringsplit;
    boost::algorithm::split(stringsplit,argv[1],boost::is_any_of("."));
    data.save_data_label(stringsplit[0]);
    return 0;
}
