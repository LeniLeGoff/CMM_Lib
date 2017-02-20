#include <iostream>
#include <fstream>
#include <memory>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <math.h>
#include <ctime>

#include <boost/random.hpp>
#include <tbb/tbb.h>

#include <iagmm/gmm.hpp>
#include <iagmm/component.hpp>

#include <boost/archive/binary_iarchive.hpp>

#include <SFML/Graphics.hpp>

#define MAX_X 100
#define MAX_Y 100
#define PI 3.14159265359
#define NBR_CLUSTER 2

using namespace iagmm;


int main(int argc, char** argv){
    srand(std::time(NULL));
    tbb::task_scheduler_init init;



    double A;
    sf::RenderWindow window(sf::VideoMode(MAX_X*4,MAX_Y*4),"dataset");

    int real_space[MAX_X][MAX_Y];
    double estimated_space[MAX_X][MAX_Y];

    GMM gmm;

    std::ifstream ifs("archive_gmm");
    boost::archive::binary_iarchive iarch(ifs);
    iarch >> gmm;

    for(const auto& comps: gmm.model())
        for(const auto& c : comps.second)
            c->print_parameters();

    std::vector<sf::RectangleShape> rects_estimated(MAX_Y*MAX_X,
                                                    sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::CircleShape> components_center;

    int coord[2];

    for(int i = 0; i < MAX_X*MAX_Y; i++){
        coord[0] = i%MAX_X;
        if(coord[0] == MAX_X - 1)
            coord[1]++;
        if(coord[1] >= MAX_Y)
            coord[1] = 0;

        estimated_space[coord[0]][coord[1]] = 0.;

        rects_estimated[i].setPosition(coord[0]*4,coord[1]*4);
        rects_estimated[i].setFillColor(sf::Color::White);


    }

    for(int i = 0; i < MAX_X; i++){
        for(int j = 0; j < MAX_Y; j++){

            double est = gmm.compute_estimation(
                        Eigen::Vector2d((double)i/(double)MAX_X,(double)j/(double)MAX_Y),1);


            rects_estimated[i + j*MAX_Y].setFillColor(
                        sf::Color(255*est,0,255*(1-est))
                        );

        }
    }


    components_center.clear();
    for(const auto& components : gmm.model()){
        for(const auto& c : components.second){
            components_center.push_back(sf::CircleShape(2.));

            components_center.back().setFillColor(sf::Color(components.first*255,255,(components.first-1)*255));
            components_center.back().setPosition(c->get_mu()(0)*MAX_X*4,c->get_mu()(1)*MAX_Y*4);
        }
    }
    std::cout << "_________________________________________________________________" << std::endl;
    std::cout << "total number of samples in the model : " << gmm.number_of_samples() << std::endl;
    std::cout << "_________________________________________________________________" << std::endl;


    while(window.isOpen()){

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
//        window.clear(sf::Color::White);


        for(auto rect: rects_estimated)
            window.draw(rect);
        for(auto circle: components_center)
            window.draw(circle);




        window.display();

    }

    return 0;
}
