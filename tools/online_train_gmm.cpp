#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <iagmm/trainer.hpp>
#include <iagmm/gmm.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>


int main(int argc, char** argv){

    if(argc =! 3){
        std::cerr << "Usage : yaml file with dataset, output file, (optional) dataset max size" << std::endl;
        return 1;
    }
    bool with_update_dataset = argc == 4;


    int dimension, nbr_class;
    iagmm::TrainingData dataset;
    dataset.load_yml(argv[1],dimension,nbr_class);


    iagmm::GMM gmm(dimension,nbr_class);
    if(with_update_dataset)
        gmm.set_dataset_size_max(std::stoi(argv[3]));

    std::stringstream ss;
    boost::filesystem::path file_path(argv[1]);
    ss << file_path.parent_path().c_str()
            << "/" << argv[2];
    if(!boost::filesystem::exists(ss.str()))
        boost::filesystem::create_directory(ss.str());

    int i;
    for(i = 0; i < dataset.size(); i++){

        std::cout << "_____________________________________" << std::endl;
        std::cout << "iteration number : " << i << std::endl;
        std::cout << "total number of samples : " << gmm.number_of_samples() << std::endl;
        std::cout << std::endl;
        gmm.add(dataset[i].second,dataset[i].first);
        gmm.update();
        if(with_update_dataset)
            gmm.update_dataset();
        std::cout << gmm.print_info() << std::endl;
        std::stringstream stream_iter;
        stream_iter << ss.str() << "/iteration_" << i;

        if(!boost::filesystem::exists(stream_iter.str()))
            boost::filesystem::create_directory(stream_iter.str());

        std::stringstream stream_dataset,stream_gmm;
        stream_dataset << stream_iter.str() << "/dataset_meanFPFHLabHist.yml";
        gmm.get_samples().save_yml(stream_dataset.str());
        stream_gmm << stream_iter.str() << "/gmm_dataset_meanFPFHLabHist";
        std::ofstream ofs(stream_gmm.str());
        boost::archive::text_oarchive toa(ofs);
        toa << gmm;
        ofs.close();
    }



    return 0;
}
