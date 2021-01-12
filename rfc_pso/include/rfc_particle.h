#ifndef _PSO_PARTICLE_H_
#define _PSO_PARTICLE_H_

#include <cstdint>
#include <ostream>
#include <istream>

#include <dlib/rand.h>
#include <dlib/matrix.h>
#include <dlib/serialize.h>


const uint64_t x1_size = 2;

// ----------------------------------------------------------------------------------------

class particle
{
private:

public:
    uint64_t number;
    uint64_t iteration;

    dlib::matrix<float, 1, x1_size> x1;

    particle() {}

    //particle(dlib::matrix<double> x_) : x(x_) {}
    particle(
        dlib::matrix<float> x1_
    ) : x1(x1_)
    {
        number = 0;
        iteration = 0;
    }

    dlib::matrix<float> get_x1() { return x1; }

    void set_number(uint64_t n) { number = n; }

    uint64_t get_number(void) { return number; }

    void set_iteration(uint64_t i) { iteration = i; }

    uint64_t get_iteration(void) { return iteration; }

    // ----------------------------------------------------------------------------------------
    // This function is used to randomly initialize 
    void rand_init(dlib::rand& rnd, std::pair<particle, particle> limits)
    {
        long idx;

        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = (float)rnd.get_double_in_range(limits.first.x1(0, idx), limits.second.x1(0, idx));
        }
    }

    // ----------------------------------------------------------------------------------------
    // This fucntion checks the particle value to ensure that the limits are not exceeded
    void limit_check(std::pair<particle, particle> limits)
    {
        long idx;

        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = std::max(std::min(limits.second.x1(0, idx), x1(0, idx)), limits.first.x1(0, idx));
        }

    }

    // ----------------------------------------------------------------------------------------
    static particle get_rand_particle(dlib::rand& rnd)
    {
        long idx;
        particle p;

        for (idx = 0; idx < p.x1.nc(); ++idx)
        {
            p.x1(0, idx) = (float)rnd.get_random_double();
        }

        return p;
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator+(const particle& p1, const particle& p2)
    {
        return particle(p1.x1 + p2.x1);
        //return particle(p1.x + p2.x);
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator-(const particle& p1, const particle& p2)
    {
        return particle(p1.x1 - p2.x1);
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator*(const particle& p1, const particle& p2)
    {
        return particle(dlib::pointwise_multiply(p1.x1, p2.x1));
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator*(const particle& p1, double& v)
    {
        return particle(v * p1.x1);
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator*(double& v, const particle& p1)
    {
        return particle(v * p1.x1);
    }

    // ----------------------------------------------------------------------------------------
    friend void serialize(const particle& item, std::ostream& out)
    {
        dlib::serialize("base_particle", out);
        dlib::serialize(item.number, out);
        dlib::serialize(item.iteration, out);
        dlib::serialize(item.x1, out);
    }

    // ----------------------------------------------------------------------------------------
    friend void deserialize(particle& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "base_particle")
            throw dlib::serialization_error("Unexpected version found: " + version + " while deserializing particle.");
        dlib::deserialize(item.number, in);
        dlib::deserialize(item.iteration, in);
        dlib::deserialize(item.x1, in);
    }

    // ----------------------------------------------------------------------------------------
    inline friend std::ostream& operator<< (std::ostream& out, const particle& item)
    {
        out << "x=" << dlib::csv << item.x1;
        return out;
    }
};


#endif  // _PSO_PARTICLE_H_
