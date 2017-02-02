#ifndef ORBITAL_H
#define ORBITAL_H

#include <complex>

#include "constants.h"

#include "TelePrompter.h"

template<int D> class FunctionTree;
template<int D> class FunctionSerialTree;

class Orbital {
public:
    Orbital(int occ, int s);
    Orbital(const Orbital &orb);
    Orbital &operator=(const Orbital &orb);
    virtual ~Orbital() { clear(); }
    void clear(bool free = true);

    int getNNodes() const;
    int getSpin() const { return this->spin; }
    int getOccupancy() const { return this->occupancy; }
    double getError() const { return this->error; }

    void setSpin(int s) { this->spin = s; }
    void setOccupancy(int occ) { this->occupancy = occ; }
    void setError(double err) { this->error = err; }

    bool isConverged(double prec) const;

    bool hasReal() const { if (this->real == 0) return false; return true; }
    bool hasImag() const { if (this->imag == 0) return false; return true; }

    void allocReal();
    void allocImag();

    void setReal(FunctionTree<3> *re) { this->real = re; }
    void setImag(FunctionTree<3> *im) { this->imag = im; }

    FunctionTree<3> &re() { return *this->real; }
    FunctionTree<3> &im() { return *this->imag; }

    void compare(const Orbital &orb) const;
    int compareSpin(const Orbital &orb) const;
    int compareOccupancy(const Orbital &orb) const;

    std::complex<double> dot(Orbital &ket);
    double getSquareNorm() const;
    /*! determines the exchange factor to be used in the calculation of the exact exchange
     *
     * \param [in] orb input orbital to which K is applied
     *
     * The factor is computed in terms of the occupancy of the two orbitals and in terms of the spin
     * 0.5 factors are used in order to preserve occupancy of the set of doubly occupied orbitals
     * this-> is the orbital defining the operator whereas the input orbital (orb) is the one              
     * the operator is applied to
     *
     * Occupancy: Single/Double
     * Spin: alpha/beta
     *
     * K (this->) | orb (input) | factor
     * alpha      | alpha       | 1.0       
     * alpha      | beta        | 0.0       
     * alpha      | double      | 0.5      
     * -------------------------------
     * beta       | alpha       | 0.0       
     * beta       | beta        | 1.0       
     * beta       | double      | 0.5 
     * -------------------------------
     * double     | alpha       | 1.0       
     * double     | beta        | 1.0       
     * double     | double      | 1.0       
     *
     */
    double getExchangeFactor(const Orbital &orb) const;

    void normalize();
    void orthogonalize(Orbital &orb);

    char printSpin() const {
        char sp = 'u';
        if (this->spin == Alpha) sp = 'a';
        if (this->spin == Beta) sp = 'b';
        return sp;
    }

    void send_Orbital(int dest, int tag);
    void Rcv_Orbital(int source, int tag);
    void Isend_Orbital(int dest, int tag);
    void IRcv_Orbital(int source, int tag);

    friend std::ostream& operator<<(std::ostream &o, Orbital &orb) {
        o << std::setw(25) << orb.getSquareNorm();
        o << std::setw(3) << orb.getOccupancy();
        o << std::setw(4) << orb.printSpin();
        o << std::setw(24) << orb.getError() << std::endl;
        return o;
    }

protected:
    int spin;
    int occupancy;

    double error;
    FunctionTree<3> *real;
    FunctionTree<3> *imag;
};

#endif // ORBITAL_H
