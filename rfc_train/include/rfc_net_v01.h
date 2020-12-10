#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <cstdint>

#include "dlib/matrix.h"
#include "dlib/dnn.h"

// ----------------------------- Decoder Setup --------------------------------
template <long N1, typename SUBNET>
using decoder1 = dlib::fc<N1, SUBNET>;

template <long N1, long N2, typename SUBNET>
using decoder2 = dlib::fc<N1, dlib::fc<N2, SUBNET>>;

template <long N1, long N2, long N3, typename SUBNET>
using decoder3 = dlib::fc<N1, dlib::fc<N2, dlib::fc<N3, SUBNET>>>;

// ----------------------------- Encoder Setup --------------------------------
template <long N1, typename SUBNET>
using encoder1 = dlib::fc<N1, SUBNET>;

template <long N1, long N2, typename SUBNET>
using encoder2 = dlib::fc<N1, dlib::fc<N2, SUBNET>>;

template <long N1, long N2, long N3, typename SUBNET>
using encoder3 = dlib::fc<N1, dlib::prelu<dlib::fc<N2, dlib::prelu<dlib::fc<N3, SUBNET>>>>>;

// ----------------------------------------------------------------------------
using net_type = dlib::loss_mean_squared_multioutput<

    // decoder
    //decoder1<2048, 
    decoder2<2048, 64,
    //decoder3< 1024, 256, 128,
    //dlib::prelu<

    // encoder
    //encoder<1024, 128, 128,

    //encoder1<4096,
    encoder2<64, 8192,
    //encoder3<128, 256, 256,
    //encoder3<256, 256, 256,
    //encoder3<256, 256, 256,
    //encoder3<256, 256, 4096,

    // input
    dlib::input<dlib::matrix<float>>
    > > >;

// ----------------------------------------------------------------------------
template <typename net_type>
net_type config_net(std::vector<uint32_t> params)
{
    net_type net = net_type(dlib::num_fc_outputs(params[0])
        //dlib::num_fc_outputs(params[1])
        //dlib::num_fc_outputs(params[2])
        //dlib::num_fc_outputs(params[3]),
        //dlib::num_fc_outputs(params[4]),
        //dlib::num_fc_outputs(params[5])
    );

    return net;

}   // end of config_net

// ----------------------------------------------------------------------------
using decoder_net = dlib::loss_mean_squared_multioutput<   
    decoder3<2048, 1024, 256, 
    dlib::input<dlib::matrix<float>>
    > >;

// ----------------------------------------------------------------------------
using encoder_net = 
    encoder3<16, 128, 256,
    dlib::input<dlib::matrix<float>>
    >;

#endif  // NET_DEFINITION_H
