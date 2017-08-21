#ifndef MCS_FCT_HPP
#define MCS_FCT_HPP

namespace iagmm{

typedef std::function<double(const Eigen::VectorXd&,const std::vector<double>&)> comb_fct_t;
typedef std::function<void(Eigen::VectorXd&, std::vector<Classifier::Ptr>&, const TrainingData&)> opt_fct_t;

struct combinatorial{
    static std::map<std::string,comb_fct_t> create_map(){
        std::map<std::string,comb_fct_t> map;

        map.emplace("sum",
                    [](const Eigen::VectorXd& parameters,
                    const std::vector<double>& estimations) -> double {
            double sum = 0;
            for(int i = 0; i < estimations.size(); i++){
                sum += parameters(i)*estimations[i];
            }
            sum = sum/(double)estimations.size();

            return sum;

        });
        return map;
    }
    static const std::map<std::string,comb_fct_t> fct_map;

};

struct optimization{
    static std::map<std::string,opt_fct_t> create_map(){
        std::map<std::string,comb_fct_t> map;

        map.emplace("conjugateGrad",
                    [](Eigen::VectorXd& parameters,
                    std::vector<Classifier::Ptr>& classifiers,
                    const TrainingData& samples){



        });

    }
};
}

#endif // MCS_FCT_HPP
