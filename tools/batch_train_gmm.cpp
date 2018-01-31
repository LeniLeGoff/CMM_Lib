#include <iostream>
#include <sstream>
#include <fstream>
#include <iagmm/trainer.hpp>
#include <iagmm/gmm.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>



int main(int argc, char** argv){

    if(argc =! 3){
        std::cerr << "Usage : yaml file with dataset, nbr of epoch" << std::endl;
        return 1;
    }

    int nbr_epoch = std::stoi(argv[2]);

    iagmm::Trainer<iagmm::GMM> trainer(argv[1],50);
    trainer.initialize();
    trainer.access_classifier().set_update_mode(iagmm::GMM::update_mode_t::BATCH);

    for(int i = 0; i < nbr_epoch; i++){
        std::cout << "_____________________________________" << std::endl;
        std::cout << "epoch number : " << i << std::endl;
        std::cout << std::endl;
        trainer.epoch();
        std::cout << "error : " << trainer.test() << std::endl;
    }

    boost::filesystem::path file_path(argv[1]);
    std::stringstream ss;
    ss << file_path.parent_path().c_str() << "/batch_learned_gmm";
    std::ofstream ofs(ss.str());
    boost::archive::text_oarchive toa(ofs);
    toa << trainer.access_classifier();
    ofs.close();

    return 0;
}
