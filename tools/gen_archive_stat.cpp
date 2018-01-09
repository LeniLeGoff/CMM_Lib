#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

#include <yaml-cpp/yaml.h>

#include <boost/archive/text_iarchive.hpp>

#include <iagmm/gmm.hpp>

using namespace iagmm;

void write_file(GMM& gmm, const std::string file){
    std::ofstream ofs(file);
    if(!ofs.is_open()){
        std::cerr << "impossible d'ouvrir le fichier : " << file << std::endl;
        return;
    }

    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;

    YAML::Emitter emitter;
    emitter << YAML::BeginMap; //GLOBAL MAP
    emitter << YAML::Key << "stat" << YAML::Value
            << YAML::BeginMap; //MAP STAT
    for(int c = 0; c < gmm.get_nbr_class(); c++){
        std::stringstream class_name;
        class_name << "class_" << c;
        emitter << YAML::Key << class_name.str() << YAML::Value
                     << YAML::BeginMap;//MAP CLASS
        int i = 0;
        for(const Component::Ptr& comp : gmm.model().at(c)){
            std::stringstream sstream;
            sstream << "component_" << i;
            emitter << YAML::Key << sstream.str() << YAML::Value
                    << YAML::BeginMap //MAP COMPONENT
                    << YAML::Key << "nb_samples" << YAML::Value << comp->get_samples().size()
                    << YAML::Key << "mean" << YAML::Value
                        << YAML::BeginSeq;
            for(int k = 0; k < comp->get_mu().rows(); k++){
                emitter << comp->get_mu()(k);
            }
            emitter << YAML::EndSeq;
            emitter << YAML::Key << "eigen_vectors" << YAML::Value
                    << YAML::BeginSeq;

            comp->compute_eigenvalues(eigenvalues,eigenvectors);
            for(int k = 0; k < eigenvectors.cols(); k++){
                emitter << YAML::BeginSeq;
                for(int l = 0; l < eigenvectors.rows();l++){
                    emitter << eigenvectors.col(k)(l);
                }
                emitter << YAML::EndSeq;
            }
            emitter << YAML::EndSeq;
            emitter << YAML::Key << "factor" << YAML::Value << comp->get_factor()
                    << YAML::EndMap; //END MAP COMPONENT
            i++;
        }
        emitter << YAML::EndMap; //END MAP CLASS
    }
    emitter << YAML::EndMap //END MAP STAT
            << YAML::EndMap; //END GLOBAL MAP

    ofs << emitter.c_str();
    ofs.close();
}

int main(int argc, char** argv){

    if(argc != 2){
        std::cout << "usage : file name containing gmm archive" << std::endl;
        return 1;
    }

    std::string output_file;
    output_file = std::string(argv[1]) + "_stat.yml";

    std::ifstream ifs(argv[1]);
    if(!ifs.is_open()){
        std::cerr << "impossible d'ouvrir le fichier : " << argv[1] << std::endl;
        return 1;
    }

    boost::archive::text_iarchive iarch(ifs);
    GMM gmm;
    iarch >> gmm;

    ifs.close();

    write_file(gmm,output_file);

    return 0;
}
