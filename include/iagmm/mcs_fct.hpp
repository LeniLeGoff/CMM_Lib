#ifndef MCS_FCT_HPP
#define MCS_FCT_HPP


#include <iagmm/classifier.hpp>

#include <functional>
#include <vector>
#include <map>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/IterativeLinearSolvers>

namespace iagmm{

typedef std::function<double(const Eigen::VectorXd&,
                             const std::vector<double>&)> comb_fct_t;


struct combinatorial{
    static std::map<std::string,comb_fct_t> create_map(){
        std::map<std::string,comb_fct_t> map;

        map.emplace("sum",
                    [](const Eigen::VectorXd& confidences,
                    const std::vector<double>& estimations) -> double {
            double sum = 0;
            for(int i = 0; i < estimations.size(); i++){
                sum += confidences(i)*estimations[i];
            }
            return sum;

        });

        map.emplace("avg",
                    [](const Eigen::VectorXd& confidences,
                const std::vector<double>& estimations) -> double {

            double sum = 0;
            for(int i = 0; i < estimations.size(); i++){
                sum += confidences[i]*estimations[i];
            }
            sum = sum/(double)estimations.size();

            return sum;

        });

        map.emplace("max",
                    [](const Eigen::VectorXd& confidences,
                                    const std::vector<double>& estimations) -> double {

            double max = 0, max_val = confidences(0);
            for(int i = 0; i < confidences.size(); i++){
                if(max_val < confidences[i]){
                    max = i;
                }
            }

            return estimations[max];

        });

//        map.emplace("max2",
//                    [](const Eigen::VectorXd& parameters,
//                                    const std::vector<double>& estimations) -> double {

//            Eigen::Vectpr

//            double max = 0, max_val = confidences(0);
//            for(int i = 0; i < confidences.size(); i++){
//                if(max_val < confidences[i]){
//                    max = i;
//                }
//            }

//            return estimations[max];

//        });

        return map;
    }
    static const std::map<std::string,comb_fct_t> fct_map;

};

}

#endif // MCS_FCT_HPP
