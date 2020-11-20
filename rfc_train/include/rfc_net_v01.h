#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <cstdint>

#include "dlib/matrix.h"
#include "dlib/dnn.h"

// ----------------------------- Decoder Setup --------------------------------
template <long N1, long N2, long N3, typename SUBNET>
using decoder = dlib::fc<N1, dlib::prelu<dlib::fc<N2, dlib::prelu<dlib::fc<N3, SUBNET>>>>>;

// ----------------------------- Encoder Setup --------------------------------
template <long N1, long N2, long N3, typename SUBNET>
using encoder = dlib::fc<N1, dlib::prelu<dlib::fc<N2, dlib::prelu<dlib::fc<N3, SUBNET>>>>>;

// ----------------------------------------------------------------------------
using net_type = dlib::loss_mean_squared_multioutput<

    // decoder
    decoder<2048, 16, 16, 
    
    //dlib::prelu<

    // encoder
    encoder<16, 128, 256,

    // input
    dlib::input<dlib::matrix<float>>
    > > >;

// ----------------------------------------------------------------------------
template <typename net_type>
net_type config_net(std::vector<uint32_t> params)
{
    net_type net = net_type(dlib::num_fc_outputs(params[0]),
        dlib::num_fc_outputs(params[1]),
        dlib::num_fc_outputs(params[2]),
        dlib::num_fc_outputs(params[3]),
        dlib::num_fc_outputs(params[4]),
        dlib::num_fc_outputs(params[5])
    );

    return net;

}   // end of config_net

// ----------------------------------------------------------------------------
using decoder_net = dlib::loss_mean_squared_multioutput<   
    decoder<2048, 16, 16, 
    dlib::input<dlib::matrix<float>>
    > >;

// ----------------------------------------------------------------------------
using encoder_net = 
    encoder<16, 128, 256,
    dlib::input<dlib::matrix<float>>
    >;

#endif  // NET_DEFINITION_H
