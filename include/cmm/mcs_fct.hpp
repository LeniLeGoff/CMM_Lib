#ifndef MCS_FCT_HPP
#define MCS_FCT_HPP


#include <cmm/classifier.hpp>

#include <functional>
#include <vector>
#include <map>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/IterativeLinearSolvers>

namespace cmm{

typedef std::function<std::vector<double>(const Eigen::VectorXd&,
                             const std::vector<std::vector<double>>&)> comb_fct_t;

typedef std::function<Eigen::VectorXd(const Eigen::VectorXd&)> param_fct_t;

/**
 * @brief Structure gathering all the possible combination function for the multi-classifier system.
 *  The structure currently includes :
 *      - sum : combines the predictions with a weighted sum
 *      - avg : combines the predictions computing the weighted average of them
 *      - max : return the prediction of the classifier with the highest parameter which is assumed to be a confidence.
 */
struct combinatorial{
    static std::map<std::string,comb_fct_t> create_map(){
        std::map<std::string,comb_fct_t> map;

        map.emplace("sum",
                    [](const Eigen::VectorXd& confidences,
                    const std::vector<std::vector<double>>& estimations) -> std::vector<double> {
            std::vector<double> sum(estimations[0].size(),0);
            for(int i = 0; i < estimations.size(); i++){
                for(int j = 0; j < estimations[i].size(); j++)
                    sum[j] += confidences(i)*estimations[i][j];
            }
            return sum;

        });

        map.emplace("avg",
                    [](const Eigen::VectorXd& confidences,
                const std::vector<std::vector<double>>& estimations) -> std::vector<double> {

            std::vector<double> sum(estimations[0].size(),0);
            for(int i = 0; i < estimations.size(); i++){
                for(int j = 0; j < estimations[i].size(); j++)
                    sum[j] += confidences(i)*estimations[i][j];
            }
            for(int j = 0; j < estimations[0].size(); j++)
                sum[j] = sum[j]/(double)estimations.size();

            return sum;

        });

        map.emplace("max",
                    [](const Eigen::VectorXd& confidences,
                                    const std::vector<std::vector<double>>& estimations) -> std::vector<double> {


            double max = 0, max_val = confidences(0);
            for(int i = 0; i < confidences.size(); i++){
                if(max_val < confidences[i]){
                    max = i;
                }
            }

            return estimations[max];

        });

        return map;
    }
    static const std::map<std::string,comb_fct_t> fct_map;

};

/**
 * @brief Structure gathering all the possible filter function of the parameters for the multi-classifier system.
 * The structure includes :
 *      - none : no filtering is applied
 *      - linear : center the parameters around the average and with value between 0 and 1.
 *      - sigmoid : center the parameters around the average using a sigmoïd function and with a value 0 and 1.
 */
struct param_estimation{
    static std::map<std::string,param_fct_t> create_map(){
        std::map<std::string,param_fct_t> map;

        map.emplace("none",
                    [](const Eigen::VectorXd& input) -> Eigen::VectorXd {
            return input;

        });

        map.emplace("linear",
                    [](const Eigen::VectorXd& input) -> Eigen::VectorXd {
            Eigen::VectorXd params(input.size());
            double sum = 0;
            for(int i = 0; i < input.size(); i++){
                sum += input(i);
            }
            sum = sum/(double)input.size();

            for(int i = 0; i < params.size(); i++){
                params(i) = 0.5 + input(i) - sum;
            }

            return params;

        });

        map.emplace("sigmoid",
                    [](const Eigen::VectorXd& input) -> Eigen::VectorXd {
            Eigen::VectorXd params(input.size());
            double sum = 0;
            for(int i = 0; i < input.size(); i++){
                sum += input(i);
            }
            sum = sum/(double)input.size();
            double val;
            for(int i = 0; i < params.size(); i++){
                val = 0.5 + input(i) - sum;
                params(i) =  1./(1. + exp(-100.*val - .5));
            }

            return params;

        });

        return map;
    }
    static const std::map<std::string,param_fct_t> fct_map;
};

}

#endif // MCS_FCT_HPP
