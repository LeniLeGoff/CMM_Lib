#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <math.h>
#include <tbb/tbb.h>
#include <ctime>

#include <boost/random.hpp>
#include <boost/chrono.hpp>

#include <iagmm/gmm.hpp>
#include <iagmm/component.hpp>
#include <iagmm/trainer.hpp>

#include <boost/archive/binary_oarchive.hpp>

#include <SFML/Graphics.hpp>

#define MAX_X 100
#define MAX_Y 100
#define PI 3.14159265359
#define NBR_CLUSTER 2

using namespace iagmm;

double compute_f(double A,double x, double y){
    return -(y+A)*sin(sqrt(abs(x/2+(y+A))))-x*sin(sqrt(abs(x-(y+A))));
}


int main(int argc, char** argv){
    srand(std::time(NULL));
    boost::random::mt19937 gen;
    boost::chrono::system_clock::time_point timer;

    tbb::task_scheduler_init init;

    double A;

    sf::RenderWindow window(sf::VideoMode(MAX_X*4*3,MAX_Y*4*2),"dataset");

    if(argc < 4){
        std::cerr << "usage : data set size, nbr epoch, batch size, (optional) A" << std::endl;
        return 1;
    }
    else if(argc == 5)
        A = std::stod(argv[4]);
    else
        A = rand()%100;

    int data_set_size = std::stoi(argv[1]);
    int nbr_epoch = std::stoi(argv[2]);
    int batch_size = std::stoi(argv[3]);

    std::cout << "A = " << A << std::endl;
    int real_space[MAX_X][MAX_Y];
    double estimated_space[MAX_X][MAX_Y];

    double error;


    double cumul_est;


    std::vector<sf::RectangleShape> rects_explored(MAX_Y*MAX_X,
                                                   sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_real(MAX_Y*MAX_X,
                                               sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_estimated(MAX_Y*MAX_X,
                                                    sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_exact_est(MAX_Y*MAX_X,
                                                    sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_confidence(MAX_Y*MAX_X,
                                                    sf::RectangleShape(sf::Vector2f(4,4)));

    std::vector<sf::CircleShape> components_center;

    std::vector<sf::RectangleShape> error_curve;

    Eigen::Vector2i coord(0,0);
    std::vector<std::pair<Eigen::VectorXd,double>> all_sample(MAX_Y*MAX_X);


    for(int i = 0; i < MAX_X*MAX_Y; i++){
        coord[0] = i%MAX_X;
        if(coord[0] == MAX_X - 1)
            coord[1]++;
        if(coord[1] >= MAX_Y)
            coord[1] = 0;

//        all_sample.push_back(Eigen::Vector2d(coord[0]/(double)MAX_X,coord[1]/(double)MAX_Y));

        if(compute_f(A,coord[0] - MAX_X/2,coord[1] - MAX_Y/2) > 0){
            real_space[coord[0]][coord[1]] = 1;
            rects_real[i].setFillColor(sf::Color(255,0,0));
        }else{
            real_space[coord[0]][coord[1]] = 0;
            rects_real[i].setFillColor(sf::Color(0,0,255));
        }
        estimated_space[coord[0]][coord[1]] = 0.;

        rects_estimated[i].setPosition(coord[0]*4+MAX_X*4,coord[1]*4);
        rects_estimated[i].setFillColor(sf::Color::White);
        rects_exact_est[i].setPosition(coord[0]*4+MAX_X*4*2,coord[1]*4);
        rects_exact_est[i].setFillColor(sf::Color::White);

        rects_real[i].setPosition(coord[0]*4,coord[1]*4);
        rects_explored[i].setPosition(coord[0]*4+MAX_X*4*2,coord[1]*4);
        rects_explored[i].setFillColor(sf::Color::Transparent);

        rects_confidence[i].setPosition(coord[0]*4+MAX_X*4*2,coord[1]*4+MAX_X*4);
        rects_confidence[i].setFillColor(sf::Color::White);

    }

    boost::random::uniform_real_distribution<> dist(0,1);
    float x,y;
    int k,l;
    TrainingData data_set;
    for(int i = 0; i < data_set_size; i++){
        x = dist(gen); y = dist(gen);
        k = x*MAX_X; l = y*MAX_Y;
        data_set.add(real_space[k][l],Eigen::Vector2d(x,y));
        rects_explored[k + (l)*MAX_Y].setFillColor(
                    sf::Color(255*real_space[k][l],0,255*(1-real_space[k][l]))
                );
    }

    Trainer<GMM> trainer(data_set,2,batch_size,0);
    trainer.access_classifier().set_distance_function(
        [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
        return (s1 - s2).squaredNorm();
    });
    trainer.initialize();
    trainer.access_classifier().set_update_mode(GMM::BATCH);

    int iteration = 0;

    error = 1.;
    while(window.isOpen()){
        timer  = boost::chrono::system_clock::now();
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        window.clear(sf::Color::White);

        for(auto rect: rects_estimated)
            window.draw(rect);
        for(auto rect: rects_real)
            window.draw(rect);
        for(auto rect: rects_exact_est)
            window.draw(rect);
        for(auto rect: rects_explored)
            window.draw(rect);
        for(auto rect: rects_confidence)
            window.draw(rect);
        for(auto circle: components_center)
            window.draw(circle);
        for(auto err : error_curve)
            window.draw(err);



        if(iteration <= nbr_epoch){
            trainer.epoch();
        }

        cumul_est = 0;
        tbb::parallel_for(tbb::blocked_range<size_t>(0,MAX_X*MAX_Y),
                          [&](const tbb::blocked_range<size_t>& r){
            for(int i = r.begin(); i != r.end(); ++i){
                if(i%MAX_X == MAX_X - 1)
                    coord[1]++;
                if(coord[1] >= MAX_Y)
                    coord[1] = 0;
                double est = trainer.access_classifier().compute_estimation(
                            Eigen::Vector2d((double)(i%MAX_X)/(double)MAX_X,(double)(i/MAX_X)/(double)MAX_Y),1);

                error += fabs(est-real_space[i%MAX_X][i/MAX_X]);

                rects_estimated[i].setFillColor(
                            sf::Color(255*est,0,255*(1-est))
                            );
            }
        });

        error = error/(double)(MAX_X*MAX_Y);
        sf::RectangleShape error_point(sf::Vector2f(4,4));
        error_point.setFillColor(sf::Color(0,255,0));
        error_point.setPosition(sf::Vector2f(iteration,(1-error)*400+400));
        error_curve.push_back(error_point);

        components_center.clear();
        for(const auto& components : trainer.access_classifier().model()){
            for(const auto& c : components.second){
//                std::cout << c->print_parameters();
                components_center.push_back(sf::CircleShape(5.));

                components_center.back().setFillColor(sf::Color(components.first*255,100,(components.first-1)*255));
                components_center.back().setPosition(c->get_mu()(0)*MAX_X*4+MAX_X*4,c->get_mu()(1)*MAX_Y*4);
            }
        }


        std::cout << "_________________________________________________________________" << std::endl;
        std::cout << "error : " << error << std::endl;
        std::cout << "total number of samples in the model : " << trainer.access_classifier().number_of_samples() << std::endl;
        std::cout << iteration << "-------------------------------------------------------------------" << std::endl;
        std::cout << "Time spent " << boost::chrono::duration_cast<boost::chrono::milliseconds>(boost::chrono::system_clock::now() - timer) << std::endl;
        std::cout << "_________________________________________________________________" << std::endl;

        iteration++;

        window.display();

    }

    std::ofstream of("archive_gmm");
    boost::archive::binary_oarchive oarch(of);
    oarch << trainer.access_classifier();

    return 0;
}
