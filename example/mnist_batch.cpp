#include <iostream>
#include <vector>

#include "iagmm/gmm.hpp"
#include "iagmm/trainer.hpp"

#include "mnist/mnist_reader.hpp"



Eigen::VectorXd compute_histogram(const std::vector<uint8_t> &digit,int bins){
    Eigen::VectorXd hist(bins);

    int square_size = 28/std::sqrt(bins);
    int counter = 0;
    for(int j = 0; j < 28; j++){
        for(int i = 0; i < 28; i++){
            int index = i/square_size + j/square_size*28/square_size;
            hist(index) += digit[counter]/255.;
            counter++;
        }
    }
    for(int i = 0; i < hist.rows(); i++){
        hist(i) = hist(i)/(square_size*square_size);
        hist(i) = hist(i)*100;
        hist(i) = std::round(hist(i));
        hist(i) = hist(i)*0.01;
        if(hist(i) != hist(i))
            hist(i) = 0;
    }
//    for(int i = 1; i < digit.size(); i++){
//        if(i%size == 0){
//            hist(j) = counter/(255.*(float)size);
//            counter = 0;
//            j++;
//        }
//        counter+=digit[i];
//    }
    return hist;
}

int main(int argc, char** argv){

    if(argc != 5){
        std::cout << "Usage : " << std::endl;
        std::cout << "\t - location of MNIST dataset" << std::endl;
        std::cout << "\t - batch size" << std::endl;
        std::cout << "\t - number of epoch" << std::endl;
        std::cout << "\t - alpha" << std::endl;
        return 1;
    }

    std::string data_location = argv[1];
    int batch_size = std::stoi(argv[2]);
    int nbr_epoch = std::stoi(argv[3]);

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> mnist_dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(data_location);

    iagmm::Component::_outlier_thres = 0;
    iagmm::Component::_alpha = std::stod(argv[4]);
    iagmm::TrainingData train_dataset;
    iagmm::TrainingData test_dataset;
    int dimension = 49;


    for(int k = 0; k < mnist_dataset.training_images.size() ; k++)
        train_dataset.add(mnist_dataset.training_labels[k],compute_histogram(mnist_dataset.training_images[k],dimension));

    for(int k = 0; k < 1000; k++)
        test_dataset.add(mnist_dataset.test_labels[k],compute_histogram(mnist_dataset.test_images[k],dimension));


    iagmm::Trainer<iagmm::GMM> trainer(train_dataset,test_dataset,dimension,10,batch_size);
    trainer.access_classifier().set_update_mode(iagmm::GMM::BATCH);
    double error = 0;
    int i = 0;
    std::chrono::system_clock::time_point timer;

    while(i < nbr_epoch){
        std::cout << "EPOCH -- " << i << std::endl;
        trainer.epoch();
        timer  = std::chrono::system_clock::now();
        error = trainer.test();
        std::cout << "test step, time spent : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now() - timer).count() << std::endl;
        for(int i = 0; i < 10; i++)
            std::cout << "class " << i << " : " << trainer.access_classifier().model()[i].size()
                      << " : " << trainer.access_classifier().get_samples().get_data(i).size() <<  std::endl;
//        std::cout << trainer.access_classifier().print_info() << std::endl;
        std::cout << "ERROR = " << error << std::endl;
        i++;
        if(error < 0.1)
            return 0;
    }



    return 0;
}
