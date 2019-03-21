#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <cmm/trainer.hpp>
#include <cmm/gmm.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>


void usage(){
    std::cout << "Usage : " << std::endl
              << "Train a classifier from a dataset" << std::endl
              <<  "or load a trained one from an archive" << std::endl
              << " --train yaml file with dataset" << std::endl
              << " --load gmm archive file " << std::endl << std::endl
              << "Options : " << std::endl
//              << " --output output_folder" << std::endl
              << " --max_size dataset max size" << std::endl
              << " --test yaml file for test dataset " << std::endl;
}

int main(int argc, char** argv){

    if(argc < 3){
        usage();
        return 1;
    }


    bool train = false, load = false, with_update_dataset = false, test = false;
    int max_size;
    std::string test_dataset_file = "";
    std::string arg1 = argv[1];

    train = arg1 == "--train";
    load = arg1 == "--load";
    if(!train && !load) {
        usage();
        return 1;
    }


    if(argc > 4){
        std::string opt_arg = argv[3];
        with_update_dataset = opt_arg == "--max_size";
        test = opt_arg == "--test";
        if(with_update_dataset)
            max_size = std::stoi(argv[4]);
        if(test)
            test_dataset_file = argv[4];
        if(argc == 7){
            std::string opt_arg = argv[5];
            with_update_dataset = opt_arg == "--max_size";
            test = opt_arg == "--test";
            if(with_update_dataset)
                max_size = std::stoi(argv[6]);
            if(test)
                test_dataset_file = argv[6];
        }
    }


    std::cout << "with dataset update : " << with_update_dataset << std::endl
              << "test : " << test << std::endl;
    cmm::Data test_data;
    int dimension, nbr_class;

    cmm::Component::_alpha = 0.5;

    cmm::CollabMM gmm;
    gmm.set_dataset_size_max(4);
    gmm.set_loglikelihood_driver(false);
    gmm.use_confidence(true);
    gmm.use_uncertainty(true);


    if(test){
        test_data.load_yml(test_dataset_file,dimension,nbr_class);
    }



    if(train){
        cmm::Data dataset;
        dataset.load_yml(argv[2],dimension,nbr_class);
        gmm = cmm::CollabMM(dimension,nbr_class);
        if(with_update_dataset)
            gmm.set_dataset_size_max(max_size);


        std::stringstream ss;
        boost::filesystem::path file_path(argv[2]);
        ss << file_path.parent_path().c_str()
           << "/../online_retrained/";
        if(!boost::filesystem::exists(ss.str()))
            boost::filesystem::create_directory(ss.str());

        std::vector<double> conv_score;
        double llhood = 0,div_dataset;
        int diff_dataset;
        double ave_diff = 0;
        bool end_criteria = false;
        int i;
        for(i = 0; i < dataset.size(); i++){
//            if(end_criteria && i > 100)
//                break;

            gmm.add(dataset[i].second,dataset[i].first);
//            gmm._estimate_training_dataset();
            gmm.update();
//            if(with_update_dataset)
//                gmm.update_dataset();


//            llhood = gmm.loglikelihood();
            std::cout << gmm.get_samples().get_data(0).size() << " " << gmm.get_samples().get_data(1).size() << std::endl;
            diff_dataset = gmm.get_samples().get_data(0).size() - gmm.get_samples().get_data(1).size();
            div_dataset = (double)gmm.get_samples().get_data(1).size()/(double)gmm.get_samples().get_data(0).size();
            if(i == 0) ave_diff = diff_dataset;
            else ave_diff = ((i-1)*ave_diff + diff_dataset)/(float)i;
            conv_score.push_back(llhood+div_dataset);
            end_criteria =  conv_score.back() > -0.5;
            std::cout << "_____________________________________" << std::endl;
            std::cout << "iteration number : " << i << std::endl;
            std::cout << "loglikelihood : " << llhood << " diff_dataset : " << diff_dataset << std::endl;
            std::cout << "average " << ave_diff << std::endl;
            std::cout << "conv score " << conv_score.back() << std::endl;
            std::cout << "total number of samples : " << gmm.number_of_samples() << std::endl;
            std::cout << std::endl;

            //        std::cout << gmm.print_info() << std::endl;
            std::stringstream stream_iter;
            stream_iter << ss.str() << "/iteration_" << i;

            if(!boost::filesystem::exists(stream_iter.str()))
                boost::filesystem::create_directory(stream_iter.str());

            std::stringstream stream_dataset,stream_gmm;
            stream_dataset << stream_iter.str() << "/dataset_centralFPFHLabHist.yml";
            gmm.get_samples().save_yml(stream_dataset.str());
            stream_gmm << stream_iter.str() << "/gmm_archive_centralFPFHLabHist";
            std::ofstream ofs(stream_gmm.str());
            boost::archive::text_oarchive toa(ofs);
            toa << gmm;
            ofs.close();
            if(test){
                std::vector<std::vector<double>> results;
                double error = gmm.predict(test_data,results);
                double exact_err = 0;
                for(int i = 0; i < test_data.size(); i++){
                    double res = 0;
                    if(results[i][test_data[i].first] >= 1./(double)gmm.get_nbr_class())
                        res = 1;
                    exact_err += 1 - res;
                }
                exact_err = exact_err/(double)test_data.size();
                std::cout << i << " : " << exact_err << std::endl;
                //                std::cout << "error on test data : " << std::endl;
//                std::cout << " proba prediction  :" << error << std::endl;
//                std::cout << " exact prediction  :" << exact_err << std::endl;
            }
        }
        int min_i = 0;
        double max_score = conv_score[0];
        for(int i = 1; i < conv_score.size(); i++){

            if(max_score < conv_score[i]){
                max_score = conv_score[i];
                min_i = i;
            }
        }
        std::cout << "best iteration : " << min_i << " with score : " << max_score << std::endl;

    }



    if(load){
        std::ifstream ifs(argv[2]);
        boost::archive::text_iarchive iarch(ifs);
        iarch >> gmm;
//        std::cout << gmm.print_info() << std::endl;
        if(test){
            std::vector<std::vector<double>> results;
            double error = gmm.predict(test_data,results);
            double exact_err = 0;
            for(int i = 0; i < test_data.size(); i++){
                double res = 0;
                if(results[i][test_data[i].first] >= 1./(double)gmm.get_nbr_class())
                    res = 1;
                exact_err += 1 - res;
            }
            exact_err = exact_err/(double)test_data.size();
            std::cout << "error on test data : " << std::endl;
            std::cout << " proba prediction  :" << error << std::endl;
            std::cout << " exact prediction  :" << exact_err << std::endl;
        }

    }



    return 0;
}

