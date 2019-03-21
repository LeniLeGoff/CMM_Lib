#include <iostream>
#include <yaml-cpp/yaml.h>
#include <cmm/data.hpp>
#include <boost/algorithm/string.hpp>

int main(int argc, char** argv){
    if(argc < 2){
        std::cout << "Usage : path to a yml file" << std::endl;
        return 1;
    }

    cmm::Data data;
    int dimension, nbr_class;
    data.load_yml(argv[1],dimension,nbr_class);
    std::vector<std::string> stringsplit;
    boost::algorithm::split(stringsplit,argv[1],boost::is_any_of("."));
    std::string output_file = stringsplit[0] + std::string(".libsvm");
    data.save_libsvm(output_file);
    return 0;
}
