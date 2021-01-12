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


// dlib includes
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/rand.h>

//#include "copy_dlib_net.h"
#include "rfc_particle.h"
#include "pso.h"

//----------------------------------------------------------------------------------
std::string platform;

std::string version;
std::string net_name = "rfc_net_";
std::string net_sync_name = "rfc_sync_";
std::string logfile_name = "rfc_";

const uint32_t input_size = 2;
const uint32_t output_size = 8;

dlib::matrix<int16_t> output_data;
dlib::matrix<float> A, B;


//----------------------------------------------------------------------------------
double eval_pso(particle p)
{
    double result = 0.0;
    
    dlib::matrix<float> OD = p.get_x1() * A * B;

    dlib::matrix<double> error = dlib::abs(dlib::matrix_cast<double>(output_data) - dlib::matrix_cast<double>(dlib::floor(OD + 0.5)));

    result = dlib::sum(error);

    return result;

}


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
    uint32_t bp = 0;

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


    dlib::rand rnd;
    rnd = dlib::rand(124356);

    dlib::rand rnd1;
    //rnd1 = dlib::rand(0);

    // ----------------------------------------------------------------------------------------   
    //if (argc == 1)
    //{
    //    print_usage();
    //    std::cin.ignore();
    //    return 0;
    //}

    //std::string parse_filename = argv[1];

    // parse through the supplied input file
    //parse_input_file(parse_filename, version, gpu, stop_criteria, tp, train_inputfile, iq_min_max, filter_num, save_directory);

    //std::string net_name = "mnist_net_" + version;
    //std::string net_sync_name = "mnist_sync_" + version;

    // check the platform
    get_platform_control();

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
    save_directory = program_root + "results/";
    //sync_save_location = program_root + "nets/";

#else    
    // Ubuntu
    program_root = get_ubuntu_path();

    save_directory = program_root + "results/";
    sync_save_location = program_root + "nets/";

#endif

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "program root:   " << program_root << std::endl;
    //std::cout << "data directory: " << data_directory << std::endl;
    //std::cout << "net directory:  " << sync_save_location << std::endl;
    std::cout << "save directory: " << save_directory << std::endl;
    std::cout << std::endl;
    

    try
    {

        get_current_time(sdate, stime);
        logfile_name = logfile_name + sdate + "_" + stime + ".txt";

        std::cout << "Log File: " << (save_directory + logfile_name) << std::endl;
        data_log_stream.open((save_directory + logfile_name), ios::out | ios::app);

        // Add the date and time to the start of the log file
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;

        // generate some random data
        output_data = dlib::matrix_cast<int16_t>(4096 * (dlib::randm(1, output_size, rnd) - 0.5));
        data_log_stream << dlib::csv << output_data << std::endl;

        //-----------------------------------------------------------------------------
        uint32_t N = 2000;
        long max_iterations = 1600;
        double c1 = 2.4;
        double c2 = 2.1;
        double w = 1.0;
        uint32_t mode = 1;

        dlib::pso_options options(N, max_iterations, c1, c2, w, mode);

        dlib::matrix<float, 1, input_size> x1, v1;
        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = 10000.0;
            v1(0, idx) = 1.0;
        }

        std::pair<particle, particle> x_lim(particle(-x1), particle(x1));
        std::pair<particle, particle> v_lim(particle(-v1), particle(v1));

        for (idx = 0; idx < 1000; ++idx)
        {
            rnd1 = dlib::rand(idx);

            // generate the random A and B matricies that will be used to compress the data
            A = dlib::matrix_cast<float>(2.0 * (dlib::randm(input_size, 16, rnd1) - 0.5));
            B = dlib::matrix_cast<float>(2.0 * (dlib::randm(16, output_size, rnd1) - 0.5));

            std::cout << "------------------------------------------------------------------" << std::endl;

            dlib::pso<particle> p(options);

            p.init(x_lim, v_lim);

            start_time = chrono::system_clock::now();

            p.run(eval_pso);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "PSO (" << elapsed_time.count() << ")" << std::endl << std::endl;

            dlib::matrix<int16_t> OD = dlib::matrix_cast<int16_t>(dlib::floor(p.G.get_x1() * A * B + 0.5));
            std::cout << num2str(idx, "%05d: ") << "G Best = " << p.get_gbest() << ", " << p.G;
            std::cout << dlib::csv << output_data << std::endl;
            std::cout << dlib::csv << OD << std::endl;

            //data_log_stream << "------------------------------------------------------------------" << std::endl;
            data_log_stream << num2str(idx, "%05d: ,");
            data_log_stream << p.get_gbest() << ", " << p.G;
            //data_log_stream << "Output: " << dlib::csv << OD << std::endl;

        }

        bp = 4;

    }
    catch(std::exception& e)
    {
        std::cout << std::endl << e.what() << std::endl;
        std::cin.ignore();
    }

    data_log_stream.close();

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;
    
}   // end of main
