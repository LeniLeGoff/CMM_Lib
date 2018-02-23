#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

#include <yaml-cpp/yaml.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <iagmm/gmm.hpp>

using namespace iagmm;

void write_file(GMM& gmm, const std::string file,int iteration){
    std::ofstream ofs(file, std::ofstream::out | std::ofstream::app);
    if(!ofs.is_open()){
        std::cerr << "impossible d'ouvrir le fichier : " << file << std::endl;
        return;
    }

    gmm._estimate_training_dataset();

    ofs << iteration << " : " << gmm.loglikelihood() << "\n";
    ofs.close();
}

int main(int argc, char** argv){

    if(argc != 3){
        std::cout << "usage : folder of experiment archive, dataset file" << std::endl;
        return 1;
    }

    std::string output_file;
    output_file = std::string(argv[1]) + "/loglikelihood.dat";


    boost::filesystem::directory_iterator dir_it(argv[1]);
    boost::filesystem::directory_iterator end_it;
    std::vector<std::string> split_str;
    std::string gmm_path;
    std::string dataset_path = argv[2];
    bool empty_ite = false;
    TrainingData data;
    int dim, nbr_class;
    data.load_yml(dataset_path,dim,nbr_class);
    for(;dir_it != end_it; ++dir_it){
        if(!boost::filesystem::is_directory(dir_it->path().string()))
            continue;
        boost::filesystem::directory_iterator dir_it_2(dir_it->path().string());

        for(;dir_it_2 != end_it; ++dir_it_2){
            if(boost::filesystem::is_empty(dir_it_2->path())){
                empty_ite = true;
                break;
            }
            boost::split(split_str,dir_it_2->path().string(),boost::is_any_of("/"));
            boost::split(split_str,split_str.back(),boost::is_any_of("_"));
            if(split_str[0] == "gmm")
                gmm_path = dir_it_2->path().string();
            if(split_str[0] == "dataset")
                dataset_path = dir_it_2->path().string();

        }

        if(empty_ite){
            empty_ite = false;
            continue;
        }
        std::ifstream ifs(gmm_path);
        if(!ifs.is_open()){
            std::cerr << "impossible d'ouvrir le fichier : " << gmm_path << std::endl;
            return 1;
        }

        boost::archive::text_iarchive iarch(ifs);
        GMM gmm;
        iarch >> gmm;
        ifs.close();

        TrainingData data2;
        int dim, nbr_class;
        data2.load_yml(dataset_path,dim,nbr_class);

        gmm.set_samples(data);

        write_file(gmm,output_file, data2.size());

    }


    return 0;
}
