#include <iostream>
#include <vector>

#include "iagmm/gmm.hpp"
#include "iagmm/trainer.hpp"

#include "mnist/mnist_reader.hpp"



Eigen::VectorXd compute_histogram(const std::vector<uint8_t> &digit,int bins){
    Eigen::VectorXd hist(bins);

    int counter =  digit[0];
    int j = 0;
    for(int i = 1; i < digit.size(); i++){
        if(i%49 == 0){
            hist(j) = counter/(255.*49.);
            counter = 0;
            j++;
        }
        counter+=digit[i];
    }

    return hist;
}

int main(int argc, char** argv){

    if(argc != 6){
        std::cout << "Usage : " << std::endl;
        std::cout << "\t - location of MNIST dataset" << std::endl;
        std::cout << "\t - batch size" << std::endl;
        std::cout << "\t - number of epoch" << std::endl;
        std::cout << "\t - budget of training samples (max 60000)" << std::endl;
        std::cout << "\t - alpha" << std::endl;
        return 1;
    }

    std::string data_location = argv[1];
    int batch_size = std::stoi(argv[2]);
    int nbr_epoch = std::stoi(argv[3]);
    int budget = std::stoi(argv[4]);

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> mnist_dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(data_location);

    iagmm::Component::_outlier_thres = 0;
    iagmm::Component::_alpha = std::stod(argv[5]);
    iagmm::TrainingData train_dataset;
    iagmm::TrainingData test_dataset;


    for(int k = 0; k < budget; k++)
        train_dataset.add(mnist_dataset.training_labels[k],compute_histogram(mnist_dataset.training_images[k],16));

    for(int k = 0; k < mnist_dataset.test_images.size(); k++)
        test_dataset.add(mnist_dataset.test_labels[k],compute_histogram(mnist_dataset.test_images[k],16));



    iagmm::Trainer<iagmm::GMM> trainer(train_dataset,test_dataset,16,10,batch_size);
    trainer.access_classifier().set_update_mode(iagmm::GMM::BATCH);

    double error = 0;
    int i = 0;
    while(i < nbr_epoch){
        std::cout << "EPOCH -- " << i << std::endl;
        trainer.epoch();
        error = trainer.test();
        for(int i = 0; i < 10; i++)
            std::cout << "class " << i << " : " << trainer.access_classifier().model()[i].size() << std::endl;
//        std::cout << trainer.access_classifier().print_info() << std::endl;
        std::cout << "ERROR = " << error << std::endl;
        i++;
    }



    return 0;
}
