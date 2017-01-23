#include <iostream>
#include <random>
#include <memory>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <math.h>
#include <tbb/tbb.h>
#include <ctime>

#include <boost/random.hpp>

#include <iagmm/gmm.hpp>
#include <iagmm/component.hpp>

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

    tbb::task_scheduler_init init;

    double A;
    sf::RenderWindow window(sf::VideoMode(MAX_X*4*3,MAX_Y*4*2),"dataset");

    A = rand()%100;
    int real_space[MAX_X][MAX_Y];
    double estimated_space[MAX_X][MAX_Y];
    std::vector<Eigen::VectorXd> samples;
    std::vector<double> label;
    //    std::vector<Cluster::Ptr> model;
    GMM gmm;

    double error;


    std::multimap<double,Eigen::Vector2i> choice_distribution;
    double cumul_est;


    std::vector<sf::RectangleShape> rects_explored(MAX_Y*MAX_X,
                                                   sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_real(MAX_Y*MAX_X,
                                               sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_estimated(MAX_Y*MAX_X,
                                                    sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_exact_est(MAX_Y*MAX_X,
                                                    sf::RectangleShape(sf::Vector2f(4,4)));

    std::vector<sf::CircleShape> components_center;

    std::vector<sf::RectangleShape> error_curve;

    Eigen::Vector2i coord(0,0);

    for(int i = 0; i < MAX_X*MAX_Y; i++){
        coord[0] = i%MAX_X;
        if(coord[0] == MAX_X - 1)
            coord[1]++;
        if(coord[1] >= MAX_Y)
            coord[1] = 0;

        if(compute_f(A,coord[0] - MAX_X/2,coord[1] - MAX_Y/2) > 0){
            real_space[coord[0]][coord[1]] = 1;
            rects_real[i].setFillColor(sf::Color(255,0,0));
        }else{
            real_space[coord[0]][coord[1]] = -1;
            rects_real[i].setFillColor(sf::Color(0,0,255));
        }
        estimated_space[coord[0]][coord[1]] = 0.;

        rects_estimated[i].setPosition(coord[0]*4+MAX_X*4,coord[1]*4);
        rects_estimated[i].setFillColor(sf::Color::White);
        rects_exact_est[i].setPosition(coord[0]*4+MAX_X*4*2,coord[1]*4+MAX_Y*4);
        rects_exact_est[i].setFillColor(sf::Color::White);

        rects_real[i].setPosition(coord[0]*4,coord[1]*4);
        rects_explored[i].setPosition(coord[0]*4+MAX_X*4*2,coord[1]*4);

    }

    int iteration = 0;

    error = 1.;



    while(window.isOpen()){

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
        for(auto rect: rects_explored)
            window.draw(rect);
        for(auto rect: rects_exact_est)
            window.draw(rect);
        for(auto circle: components_center)
            window.draw(circle);
        for(auto err : error_curve)
            window.draw(err);




        std::vector<double> scores = gmm.model_scores();
        int k =0;
        Eigen::MatrixXd map = Eigen::MatrixXd::Zero(MAX_X,MAX_Y);
        double min, dist, cumul = 0.;
        if(!gmm.get_pos_components().empty() && !gmm.get_neg_components().empty()){
            for(int i = 0; i < MAX_X; i++){
                for(int j = 0; j < MAX_Y; j++){
                    min = (Eigen::Vector2d((double)i/100.,(double)j/100.) -
                           gmm.get_pos_components()[0]->get_mu()).squaredNorm()/(
                                gmm.get_pos_components()[0]->get_factor()*scores[0]);
                    for(const auto& comp : gmm.get_pos_components()){
                        dist = (Eigen::Vector2d((double)i/100.,(double)j/100.) -
                                comp->get_mu()).squaredNorm()/(comp->get_factor()*scores[k]);
                        k++;
                        if(min > dist)
                            min = dist;
                    }


                    for(const auto& comp : gmm.get_neg_components()){
                        dist = (Eigen::Vector2d((double)i/100.,(double)j/100.) -
                                comp->get_mu()).squaredNorm()/(comp->get_factor()*scores[k]);
                        k++;
                        if(min > dist)
                            min = dist;
                    }
                    map(i,j) = min;

                }
            }
            map = map/map.maxCoeff();
            for(int i = 0; i < MAX_X; i++){
                for(int j = 0; j < MAX_Y; j++){
                    cumul += map(i,j);
                    choice_distribution.emplace(cumul,Eigen::Vector2i(i,j));
                }
            }

            std::cout << cumul << std::endl;
            boost::random::uniform_real_distribution<> distrib(0.,cumul);
            double rand_nb = distrib(gen);
            auto it = choice_distribution.lower_bound(rand_nb);
            double val = it->first;
            std::vector<Eigen::Vector2i> possible_choice;
            while(it->first == val){
                possible_choice.push_back(it->second);
                it++;
            }

            int rnb = rand()%(possible_choice.size());
            coord[0] = possible_choice[rnb](0);
            coord[1] = possible_choice[rnb](1);
        }else{
            coord[0] = rand()%MAX_X;
            coord[1] = rand()%MAX_Y;
        }



        //        std::cout << map << std::endl;

        samples.push_back(Eigen::Vector2d((double)coord[0]/(double)MAX_X,(double)coord[1]/(double)MAX_Y));

        label.push_back(real_space[coord[0]][coord[1]]);



        rects_explored[coord[0] + (coord[1])*MAX_Y].setFillColor(
                    sf::Color(255*real_space[coord[0]][coord[1]],0,255*(1-real_space[coord[0]][coord[1]]))
                );

        gmm.update_model(samples,label);

        error = 0;
        if(samples.size() > NBR_CLUSTER){
            cumul_est = 0;
            choice_distribution.clear();
            for(int i = 0; i < MAX_X; i++){
                for(int j = 0; j < MAX_Y; j++){

                    double est = gmm.compute_GMM(Eigen::Vector2d((double)i/(double)MAX_X,(double)j/(double)MAX_Y));
                    //                    std::cout << est << std::endl;
                    //                    std::cout << (double)i/(double)MAX_X << " " << (double)j/(double)MAX_Y << std::endl;
                    //                    if(est > 1.)
                    //                        est = 1.;
                    //                    if(est < -1)
                    //                        est = -1.;

                    double dist = map(i,j);
                    //                    if(dist > 1.) dist = 1.;
                    //                    dist = dist/5.;
                    rects_exact_est[i + j*MAX_Y].setFillColor(
                                sf::Color(255*dist,
                                          255*dist,
                                          255*dist)
                                );

                    //                    if(est < 0.4)
                    //                        rects_exact_est[i + j*MAX_Y].setFillColor(
                    //                                    sf::Color(0,0,255)
                    //                                    );
                    //                    else if(est > 0.6)
                    //                        rects_exact_est[i + j*MAX_Y].setFillColor(
                    //                                    sf::Color(255,0,0)
                    //                                    );
                    //                    else rects_exact_est[i + j*MAX_Y].setFillColor(
                    //                                sf::Color(255,0,255)
                    //                                );


                    error += (est*2 - 1)*real_space[i][j] > 0 ? 0 : 1;//fabs(est-real_space[i][j]);

                    cumul_est+= 1-fabs(est);
                    choice_distribution.emplace(cumul_est,Eigen::Vector2i(i,j));

                    //                    est = (est + 1.)/2.;

                    //                    std::cout << "estimation : " << est << std::endl;

                    rects_estimated[i + j*MAX_Y].setFillColor(
                                sf::Color(255*est,0,255*(1-est))
                                );

                }
            }
        }
        error = error/(double)(MAX_X*MAX_Y);
        sf::RectangleShape error_point(sf::Vector2f(4,4));
        error_point.setFillColor(sf::Color(0,255,0));
        error_point.setPosition(sf::Vector2f(iteration,(1-error)*400+400));
        error_curve.push_back(error_point);

        //        Eigen::VectorXd eigenval;
        //        Eigen::MatrixXd eigenvect;
        components_center.clear();
        for(const auto& c : gmm.get_pos_components()){
            c->print_parameters();
            components_center.push_back(sf::CircleShape(2.));

            components_center.back().setFillColor(sf::Color(255,255,0));
            components_center.back().setPosition(c->get_mu()(0)*MAX_X*4+MAX_X*4,c->get_mu()(1)*MAX_Y*4);

            //            c->compute_eigenvalues(eigenval,eigenvect);
            //            std::cout << "[" << eigenval << "] -- [" << eigenvect << "]" << std::endl;
        }

        for(const auto& c : gmm.get_neg_components()){
            c->print_parameters();
            components_center.push_back(sf::CircleShape(2.));
            components_center.back().setFillColor(sf::Color(0,255,255));

            components_center.back().setPosition(c->get_mu()(0)*MAX_X*4+MAX_X*4,c->get_mu()(1)*MAX_Y*4);

            //            c->compute_eigenvalues(eigenval,eigenvect);
            //            std::cout << "[" << eigenval << "] -- [" << eigenvect << "]" << std::endl;
        }

        std::cout << "_________________________________________________________________" << std::endl;
        std::cout << "error : " << error << std::endl;
        std::cout << "total number of samples in the model : " << gmm.number_of_samples() << std::endl;
        std::cout << iteration << "-------------------------------------------------------------------" << std::endl;
        std::cout << "_________________________________________________________________" << std::endl;

        iteration++;

        window.display();

        //        std::cin.ignore();
    }

    return 0;
}
