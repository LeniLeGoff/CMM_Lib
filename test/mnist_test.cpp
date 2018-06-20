#include <iostream>
#include <vector>

#include "iagmm/gmm.hpp"


#include "mnist/mnist_reader.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char** argv){

    std::string data_location = argv[1];


    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(data_location);

    std::cout << sqrt(dataset.training_images[0].size()) << std::endl;
    uint8_t* data = (uint8_t*)malloc(sizeof(uint8_t)*dataset.training_images[0].size());
    for(int k = 0; k < dataset.training_images.size(); k++)
    {
        for(int i = 0; i < dataset.training_images[k].size(); i++)
            data[i] = dataset.training_images[k][i];
        cv::Mat digit_img(28,28,CV_8UC1,data);
        cv::imshow(std::to_string(dataset.training_labels[k]),digit_img);
        cv::waitKey(5);
    }


    free(data);

    return 0;
}
