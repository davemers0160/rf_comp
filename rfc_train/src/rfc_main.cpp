#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <iostream>
#include <string>

// Custom includes
#include "get_platform.h"
#include "file_ops.h"
#include "file_parser.h"
#include "get_current_time.h"
#include "num2string.h"

#include "rfc_dnn.h"
#include "rfc_net_v01.h"

// dlib includes
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/rand.h>

#include "copy_dlib_net.h"

//----------------------------------------------------------------------------------
std::string platform;

std::string version;
std::string net_name = "rfc_net_";
std::string net_sync_name = "rfc_sync_";
std::string logfile_name = "rfc_";

//----------------------------------------------------------------------------------
/*
template <typename net_type>
dlib::matrix<double, 1, 3> eval_net_performance(net_type &net, std::vector<dlib::matrix<unsigned char>> input_images, std::vector<unsigned long> input_labels)
{
    std::vector<unsigned long> predicted_labels = net(input_images);
    int num_right = 0;
    int num_wrong = 0;
    // And then let's see if it classified them correctly.
    for (size_t i = 0; i < input_images.size(); ++i)
    {
        if (predicted_labels[i] == input_labels[i])
            ++num_right;
        else
            ++num_wrong;
        
    }
    // std::cout << "training num_right: " << num_right << std::endl;
    // std::cout << "training num_wrong: " << num_wrong << std::endl;
    // std::cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << std::endl;
    
    dlib::matrix<double, 1, 3> results;
    results = (double)num_right, (double)num_wrong, (double)num_right/(double)(num_right+num_wrong);
    
    return results;
    
}   // end of eval_net_performance
*/

//----------------------------------------------------------------------------------
class visitor_copy_layer
{
public:

    visitor_copy_layer(void) {}

    template<typename input_layer_type>
    void operator()(size_t, input_layer_type&)  const
    {
        // ignore other layers
    }

    template <typename T, typename U, typename E>
    void operator()(size_t, dlib::add_layer<T, U, E>& l)  const
    {
        set_weight_decay_multiplier(l.layer_details(), new_weight_decay_multiplier);
    }

private:

    double new_weight_decay_multiplier;
};


// ----------------------------------------------------------------------------
void get_platform_control(void)
{
    get_platform(platform);

    if (platform == "")
    {
        std::cout << "No Platform could be identified... defaulting to Windows." << std::endl;
        platform = "Win";
    }

    version = version + platform;
    net_sync_name = version + "_sync";
    logfile_name = version + "_log_";
    net_name = version + "_final.dat";
}

// ----------------------------------------------------------------------------------------
void print_usage(void)
{
    std::cout << "The wrong number of parameters was entered..." << std::endl;
    std::cout << "Enter the following as arguments into the program:" << std::endl;
    std::cout << "<input_filename> " << std::endl;
    std::cout << endl;
}
//----------------------------------------------------------------------------------
int main(int argc, char** argv)
{

    uint64_t idx = 0, jdx = 0;
    uint8_t HPC = 0;
    std::string sdate, stime;

    // data IO variables
    const std::string os_file_sep = "/";
    std::string program_root;
    std::string save_directory;
    std::string sync_save_location;
    std::string train_inputfile;
    std::string train_data_directory;

    std::ofstream data_log_stream;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    // training variables
    int32_t stop = -1;
    std::vector<std::string> stop_codes = { "Minimum Learning Rate Reached.", "Max Training Time Reached", "Max Training Steps Reached" };
    std::vector<double> stop_criteria;

    std::vector<int> gpu;

    training_params tp;
    std::vector<uint32_t> filter_num;
    std::pair<int32_t, int32_t> iq_min_max;  // min_target_size, max_target_size  

    uint64_t one_step_calls = 0;
    uint64_t test_step_count = 2000;

    dlib::rand rnd;
    rnd = dlib::rand(time(NULL));

    // ----------------------------------------------------------------------------------------   
    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    std::string parse_filename = argv[1];

    // parse through the supplied input file
    parse_input_file(parse_filename, version, gpu, stop_criteria, tp, train_inputfile, iq_min_max, filter_num, save_directory);

    //std::string net_name = "mnist_net_" + version;
    //std::string net_sync_name = "mnist_sync_" + version;

    // check the platform
    get_platform_control();

    // check for HPC <- set the environment variable PLATFORM to HPC
    if (platform.compare(0, 3, "HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
    save_directory = program_root + "results/";
    sync_save_location = program_root + "nets/";

#else    
    if (HPC == 1)
    {
        //HPC version
        program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
    }
    else
    {
        // Ubuntu
        program_root = get_ubuntu_path();
    }

    save_directory = program_root + "results/";
    sync_save_location = program_root + "nets/";

#endif

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "program root:   " << program_root << std::endl;
    //std::cout << "data directory: " << data_directory << std::endl;
    std::cout << "net directory:  " << sync_save_location << std::endl;
    std::cout << "save directory: " << save_directory << std::endl;
    std::cout << std::endl;
    

    try
    {
        // load the data in using the dlib built in function
        dlib::matrix<float> train_data(2048, 1), train_label(2048, 1);
        dlib::matrix<float> td = dlib::matrix_cast<float>(dlib::randm(2048, 1));

        float r;
        for (idx = 0; idx < 2048; ++idx)
        {
            //r = 2 * rnd.get_random_float() - 1;
            r = (float)rnd.get_integer_in_range(-2048, 2048);
            train_data(idx, 0) = r;
            train_label(idx, 0) = r;
        }

        std::vector< dlib::matrix<float>> tr_data, tr_labels;
        for (idx = 0; idx < 50; ++idx)
        {       
            tr_data.push_back(train_data);
            tr_labels.push_back(train_label);
        }

        //auto test = std::distance(tr_data, tr_labels);

        std::cout << "------------------------------------------------------------------" << std::endl;
        //std::cout << "Loaded " << training_images.size() << " training images." << std::endl;
        
        get_current_time(sdate, stime);
        logfile_name = logfile_name + sdate + "_" + stime + ".txt";

        std::cout << "Log File: " << (save_directory + logfile_name) << std::endl;
        data_log_stream.open((save_directory + logfile_name), ios::out | ios::app);

        // Add the date and time to the start of the log file
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;
         
        //-----------------------------------------------------------------------------
        // Setup the network
        //-----------------------------------------------------------------------------

        // this sets th GPUs to use algorithms that are smaller in memory but may take a little longer to execute
        dlib::set_dnn_prefer_smallest_algorithms();

        // set the cuda device explicitly
        if (gpu.size() == 1)
            dlib::cuda::set_device(gpu[0]);

        net_type net = config_net<net_type>(filter_num);

        // configure the trainer
        dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(0.0001, 0.9, 0.99), gpu);
        trainer.set_learning_rate(tp.intial_learning_rate);
        trainer.be_verbose();
        trainer.set_synchronization_file((sync_save_location + net_sync_name), std::chrono::minutes(2));
        trainer.set_iterations_without_progress_threshold(tp.steps_wo_progess);
        trainer.set_test_iterations_without_progress_threshold(5000);
        trainer.set_learning_rate_shrink_factor(tp.learning_rate_shrink_factor);
        trainer.be_verbose();
        
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << trainer << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        data_log_stream << trainer << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;

        std::cout << "Net Name: " << net_name << std::endl;
        std::cout << net << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Net Name: " << net_name << std::endl;
        data_log_stream << net << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;


        //-----------------------------------------------------------------------------
        // TRAINING START
        //-----------------------------------------------------------------------------

        //std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Starting Training..." << std::endl;
        start_time = chrono::system_clock::now();

        while (stop < 0)
        {
            // first check to make sure that the final_learning_rate hasn't been exceeded
            if (trainer.get_learning_rate() >= tp.final_learning_rate)
            {

                trainer.train_one_step(tr_data.begin(), tr_data.end(), tr_labels.begin());

            }
            else
            {
                stop = 0;
            }

            one_step_calls = trainer.get_train_one_step_calls();

            if ((one_step_calls % test_step_count) == 0)
            {
                // this is where we will perform any needed evaluations of the network
                // detction_accuracy, correct_hits, false_positives, missing_detections

                data_log_stream << std::setw(6) << std::setfill('0') << one_step_calls << ", " << std::fixed << std::setprecision(9) << trainer.get_learning_rate() << ", ";
                data_log_stream << std::setprecision(5) << trainer.get_average_loss() << ", " << trainer.get_average_test_loss() << std::endl;

            }

            // now check to see if we've trained long enough according to the input time limit
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            if ((double)elapsed_time.count() / (double)3600.0 > stop_criteria[0])
            {
                stop = 1;
            }

            // finally check to see if we've exceeded the max number of one step training calls
            // according to the input file
            if (one_step_calls >= stop_criteria[1])
            {
                stop = 2;
            }

        }   // end of while(stop<0)

        //-----------------------------------------------------------------------------
        // TRAINING STOP
        //-----------------------------------------------------------------------------

        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        // wait for training threads to stop
        trainer.get_net();

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "Elapsed Training Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        std::cout << "Stop Code: " << stop_codes[stop] << std::endl;
        std::cout << "Final Average Loss: " << trainer.get_average_loss() << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Elapsed Training Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        data_log_stream << "Stop Code: " << stop_codes[stop] << std::endl;
        data_log_stream << "Final Average Loss: " << trainer.get_average_loss() << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl << std::endl;

        // Save the network to disk
        net.clean();
        dlib::serialize(sync_save_location + net_name) << net;


        auto res = net(train_data);

        auto r2 = res - train_data;

        auto s1 = sum(r2);

        std::cout << "sum: " << s1 << std::endl;

        auto& layer_output = dlib::layer<6>(net).get_output();
        const float* data = layer_output.host();

        dlib::matrix<float> id1 = dlib::mat<float>(data, 1, 16);

        int bp = 0;

        decoder_net dn;
        std::cout << dn << std::endl;

        encoder_net en;
        std::cout << en << std::endl;
        dlib::copy_net<6, 11, 0>(net, en);

        auto &res3_en = en(train_data);

        auto& lo_en = dlib::layer<1>(en).get_output();
        const float* data_en = lo_en.host();

        auto res_dn = dn(id1);
        dlib::copy_net<0, 5, 0>(net, dn);

        //dlib::visit_layers_range<0, 2>(net, visitor_weight_decay_multiplier(1));


        std::cout << dn << std::endl;


        //dlib::matrix<float> id2 = dlib::mat<float>(data, 16, 1);

        std::vector<dlib::matrix<float>> minibatch(1, id1);
        dlib::resizable_tensor in1;
        dn.to_tensor(minibatch.begin(), minibatch.end(), in1);

        //auto res2 = dn(id1);
        auto res3 = dn(id1);


        dlib::net_to_xml(net, save_directory + "net1.xml");
        dlib::net_to_xml(dn, save_directory + "dn.xml");

        std::cout << "number of network parameters: " << dlib::count_parameters(dn) << std::endl;

        bp = 2;


    }
    catch(std::exception& e)
    {
        std::cout << std::endl << e.what() << std::endl;
        std::cin.ignore();
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;
    
}   // end of main
