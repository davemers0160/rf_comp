#ifndef RFC_DNN_H_
#define RFC_DNN_H_

#include <cstdint>
#include <tuple>

// custom includes
#include "file_parser.h"

// ----------------------------------------------------------------------------------------

typedef struct training_params {

    training_params() = default;
    
    training_params(double ilr, double flr, double lrsf, uint32_t step) : 
        intial_learning_rate(ilr), final_learning_rate(flr), learning_rate_shrink_factor(lrsf), steps_wo_progess(step){}

    double intial_learning_rate;
    double final_learning_rate;
    double learning_rate_shrink_factor;
    uint32_t steps_wo_progess;

} training_params;

// ----------------------------------------------------------------------------------------
void parse_input_file(std::string parse_filename, 
    std::string &version, 
    std::vector<int32_t> &gpu, 
    std::vector<double> &stop_criteria, 
    training_params &tp, 
    std::string &train_input,
    std::pair<int32_t, int32_t> &iq_min_max, 
    std::vector<uint32_t> &filter_num,
    std::string &save_directory
)
{

    std::vector<std::vector<std::string>> params;
    parse_csv_file(parse_filename, params);

    for (uint64_t idx = 0; idx<params.size(); ++idx)
    {
        switch (idx)
        {

            // get the version name of the network - used for naming various files
            case 0:
                version = params[idx][0];
                break;

            // select which gpu to use
		    case 1:
                try {
                    gpu.clear();
                    for (int jdx = 0; jdx < params[idx].size(); ++jdx)
                    {
                        gpu.push_back(std::stol(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    gpu.clear();
                    gpu.push_back(0);
                }
			    break;

            // get the stopping criteria: max hours, max training steps
            case 2:
                try {

                    stop_criteria.clear();
                    for (uint64_t jdx = 0; jdx<params[idx].size(); ++jdx)
                    {
                        stop_criteria.push_back(stod(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    stop_criteria.push_back(160.0);
                    stop_criteria.push_back(250000.0);
                    std::cout << "Error getting stopping criteria.  Setting values to default." << std::endl;
                }
                break;

            // get the training parameters
            case 3:
                try {
                    tp = training_params(std::stod(params[idx][0]), std::stod(params[idx][1]), std::stod(params[idx][2]), std::stol(params[idx][3]));
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    std::cout << "Using default training parameters..." << std::endl;
                    tp = training_params(0.001, 0.000001, 0.1, 2500);
                }
                break;

            // get the file that contains the training data
            case 4:
                try {
                    train_input = params[idx][0];
                }
                catch (std::exception & e) {
                    std::cout << e.what() << std::endl;
                }
                break;

            // get the IQ min/max values
            case 5:
                try {
                    iq_min_max = std::make_pair(std::stoi(params[idx][0]), std::stoi(params[idx][1]));
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    iq_min_max = std::make_pair(-2048, 2047);
                }
                break;

            // get the number of filters for each layer
            case 6:
                try {
                    filter_num.clear();
                    for (int jdx = 0; jdx<params[idx].size(); ++jdx)
                    {
                        filter_num.push_back(std::stol(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    filter_num.clear();

                    std::cout << "Error getting filter numbers.  No values passed on." << std::endl;
                }
                break;

            case 7:
                save_directory = params[idx][0];
                break;

            default:
                break;
        }   // end of switch

    }   // end of for

}   // end of parse_input_file

#endif  // OBJ_DET_DNN_H_
