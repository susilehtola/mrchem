#include "ExchangePotential.h"
#include "OrbitalAdder.h"
#include "OrbitalMultiplier.h"
#include "PoissonOperator.h"
#include "MWConvolution.h"

using namespace std;
using namespace Eigen;

ExchangePotential::ExchangePotential(PoissonOperator &P,
                                     OrbitalVector &phi,
                                     double x_fac)
        : ExchangeOperator(P, phi, x_fac),
          exchange(phi) {
}

void ExchangePotential::setup(double prec) {
    QMOperator::setup(prec);
    calcInternalExchange();
}

void ExchangePotential::clear() {
    this->exchange.clear();
    QMOperator::clear();
}

void ExchangePotential::rotate(MatrixXd &U) {
    // negative prec means precomputed exchange is cleared
    if (this->apply_prec > 0.0) {
        OrbitalAdder add(this->apply_prec, this->max_scale);
        add.rotate(exchange, U);
    }
}

Orbital* ExchangePotential::operator() (Orbital &phi_p) {
    Orbital *preOrb = testPreComputed(phi_p);
    if (preOrb != 0) {
        println(1, "Precomputed exchange");
        return preOrb;
    }
    return calcExchange(phi_p);
}

Orbital* ExchangePotential::adjoint(Orbital &phi_p) {
    NOT_IMPLEMENTED_ABORT;
}

/** Compute exchange on the fly */
Orbital* ExchangePotential::calcExchange(Orbital &phi_p) {
    Timer timer;

    OrbitalAdder add(this->apply_prec, this->max_scale);
    OrbitalMultiplier mult(this->apply_prec, this->max_scale);
    MWConvolution<3> apply(this->apply_prec, this->max_scale);
    PoissonOperator &P = *this->poisson;

    int maxNodes = 0;
    vector<complex<double> > coef_vec;
    vector<Orbital *> orb_vec;
    int nOrbs = this->orbitals->size();
    for (int i = 0; i < nOrbs; i++) {
        Orbital &phi_i = this->orbitals->getOrbital(i);

        Orbital *phi_ip = new Orbital(phi_p);
        { // compute phi_ip = phi_i^dag * phi_p
            Timer timer;
            mult.adjoint(*phi_ip, 1.0, phi_i, phi_p);
            timer.stop();
            int nNodes = phi_ip->getNNodes();
            double time = timer.getWallTime();
            maxNodes = max(maxNodes, nNodes);
            TelePrompter::printTree(2, "Multiply", nNodes, time);
        }

        Orbital *V_ip = new Orbital(phi_p);
        { // compute V_ip = P[phi_ip]
            Timer timer;
            if (phi_ip->hasReal()) {
                V_ip->allocReal();
                apply(V_ip->real(), P, phi_ip->real());
            }
            if (phi_ip->hasImag()) {
                V_ip->allocImag();
                apply(V_ip->imag(), P, phi_ip->imag());
            }
            timer.stop();
            int nNodes = V_ip->getNNodes();
            double time = timer.getWallTime();
            maxNodes = max(maxNodes, nNodes);
            TelePrompter::printTree(2, "Applying Poisson", nNodes, time);
        }
        delete phi_ip;

        Orbital *phi_iip = new Orbital(phi_p);
        { // compute phi_iij = phi_i * V_ip
            Timer timer;
            double fac = -(this->x_factor/phi_i.getSquareNorm());
            mult(*phi_iip, fac, phi_i, *V_ip);
            timer.stop();
            int nNodes = phi_iip->getNNodes();
            double time = timer.getWallTime();
            maxNodes = max(maxNodes, nNodes);
            TelePrompter::printTree(2, "Multiply", nNodes, time);
        }
        delete V_ip;

        coef_vec.push_back(1.0);
        orb_vec.push_back(phi_iip);
    }
    Orbital *ex_p = new Orbital(phi_p);
    add(*ex_p, coef_vec, orb_vec, true);

    for (int i = 0; i < orb_vec.size(); i++) delete orb_vec[i];
    orb_vec.clear();

    timer.stop();
    double nNodes = ex_p->getNNodes();
    double time = timer.getWallTime();
    TelePrompter::printTree(1, "Applied exchange", nNodes, time);
    return ex_p;
}

/** Precompute the internal exchange */
void ExchangePotential::calcInternalExchange() {
    Timer timer;

    int nNodes = 0;
    int maxNodes = 0;

    int nOrbs = this->orbitals->size();
    for (int i = 0; i < nOrbs; i++) {
        nNodes = calcInternal(i);
        maxNodes = max(maxNodes, nNodes);
        for (int j = 0; j < i; j++) {
            nNodes = calcInternal(i,j);
            maxNodes = max(maxNodes, nNodes);
        }
    }

    for (int i = 0; i < nOrbs; i++) {
        Orbital &ex_i = this->exchange.getOrbital(i);
        this->tot_norms(i) = sqrt(ex_i.getSquareNorm());
    }

    timer.stop();
    double time = timer.getWallTime();
    TelePrompter::printTree(0, "Hartree-Fock exchange", maxNodes, time);
}

int ExchangePotential::calcInternal(int i) {
    OrbitalAdder add(-1.0, this->max_scale);
    OrbitalMultiplier mult(-1.0, this->max_scale);
    MWConvolution<3> apply(-1.0, this->max_scale);

    PoissonOperator &P = *this->poisson;
    Orbital &phi_i = this->orbitals->getOrbital(i);

    int maxNodes = 0;
    double prec = getScaledPrecision(i, i);
    prec = min(prec, 1.0e-1);
    println(2, "\n [" << i << "," << i << "]");

    Orbital *phi_ii = new Orbital(phi_i);
    { // compute phi_ii = phi_i^dag * phi_i
        Timer timer;
        mult.setPrecision(prec);
        mult.adjoint(*phi_ii, 1.0, phi_i, phi_i);
        timer.stop();
        int nNodes = phi_ii->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Multiply", nNodes, time);
    }

    Orbital *V_ii = new Orbital(phi_i);
    { // compute V_ii = P[phi_ii]
        Timer timer;
        apply.setPrecision(prec);
        if (phi_ii->hasReal()) {
            V_ii->allocReal();
            apply(V_ii->real(), P, phi_ii->real());
        }
        if (phi_ii->hasImag()) {
            V_ii->allocImag();
            apply(V_ii->imag(), P, phi_ii->imag());
        }
        timer.stop();
        int nNodes = V_ii->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Applying Poisson", nNodes, time);
    }
    if (phi_ii != 0) delete phi_ii;

    Orbital *phi_iii = new Orbital(phi_i);
    { // compute phi_iii = phi_i * V_ii
        Timer timer;
        double fac = -(this->x_factor/phi_i.getSquareNorm());
        mult(*phi_iii, fac, phi_i, *V_ii);
        this->part_norms(i,i) = sqrt(phi_iii->getSquareNorm());
        timer.stop();
        int nNodes = phi_iii->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Multiply", nNodes, time);
    }
    if (V_ii != 0) delete V_ii;

    { // compute x_i += phi_iii
        Timer timer;
        Orbital &ex_i = this->exchange.getOrbital(i);
        add.inPlace(ex_i, 1.0, *phi_iii);
        timer.stop();
        int nNodes = ex_i.getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Adding exchange", nNodes, time);
    }
    if (phi_iii != 0) delete phi_iii;
    return maxNodes;
}

int ExchangePotential::calcInternal(int i, int j) {
    OrbitalAdder add(this->apply_prec, this->max_scale);
    OrbitalMultiplier mult(this->apply_prec, this->max_scale);
    MWConvolution<3> apply(-1.0, this->max_scale);

    PoissonOperator &P = *this->poisson;
    Orbital &phi_i = this->orbitals->getOrbital(i);
    Orbital &phi_j = this->orbitals->getOrbital(j);

    int maxNodes = 0;
    double i_factor = phi_i.getExchangeFactor(phi_j);
    double j_factor = phi_j.getExchangeFactor(phi_i);
    if (i_factor < MachineZero or j_factor < MachineZero) {
        this->part_norms(i,j) = 0.0;
        return 0;
    }

    printout(2, "\n [" << i << "," << j << "]");
    printout(2, "   [" << j << "," << i << "]" << endl);
    double prec = getScaledPrecision(i, j);
    if (prec > 1.0e00) return 0;

    Orbital *phi_ij = new Orbital(phi_i);
    { // compute phi_ij = phi_i^dag * phi_j
        Timer timer;
        if (phi_i.hasImag()) NOT_IMPLEMENTED_ABORT;
        mult.adjoint(*phi_ij, 1.0, phi_i, phi_j);
        timer.stop();
        int nNodes = phi_ij->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Multiply", nNodes, time);
    }

    Orbital *V_ij = new Orbital(phi_i);
    { // compute V_ij = P[phi_ij]
        Timer timer;
        prec = min(prec, 1.0e-1);
        apply.setPrecision(prec);
        if (phi_ij->hasReal()) {
            V_ij->allocReal();
            apply(V_ij->real(), P, phi_ij->real());
        }
        if (phi_ij->hasImag()) {
            V_ij->allocImag();
            apply(V_ij->imag(), P, phi_ij->imag());
        }
        timer.stop();
        int nNodes = V_ij->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Applying Poisson", nNodes, time);
    }
    if (phi_ij != 0) delete phi_ij;

    Orbital *phi_iij = new Orbital(phi_i);
    { // compute phi_iij = phi_i * V_ij
        Timer timer;
        double fac = -(this->x_factor/phi_i.getSquareNorm());
        mult(*phi_iij, fac, phi_i, *V_ij);
        this->part_norms(i,j) = sqrt(phi_iij->getSquareNorm());
        timer.stop();
        int nNodes = phi_iij->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Multiply", nNodes, time);
    }

    Orbital *phi_jij = new Orbital(phi_i);
    { // compute phi_jij = phi_j * V_ij
        Timer timer;
        double fac = -(this->x_factor/phi_j.getSquareNorm());
        mult(*phi_jij, fac, phi_j, *V_ij);
        this->part_norms(j,i) = sqrt(phi_jij->getSquareNorm());
        timer.stop();
        int nNodes = phi_jij->getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Multiply", nNodes, time);
    }
    if (V_ij != 0) delete V_ij;

    { // compute x_i += phi_jij
        Timer timer;
        Orbital &ex_i = this->exchange.getOrbital(i);
        add.inPlace(ex_i, i_factor, *phi_jij);
        timer.stop();
        int nNodes = ex_i.getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Adding exchange", nNodes, time);
    }
    if (phi_jij != 0) delete phi_jij;

    { // compute x_j += phi_iij
        Timer timer;
        Orbital &ex_j = this->exchange.getOrbital(j);
        add.inPlace(ex_j, j_factor, *phi_iij);
        timer.stop();
        int nNodes = ex_j.getNNodes();
        double time = timer.getWallTime();
        maxNodes = max(maxNodes, nNodes);
        TelePrompter::printTree(2, "Adding exchange", nNodes, time);
    }
    if (phi_iij != 0) delete phi_iij;
    return maxNodes;
}

Orbital* ExchangePotential::testPreComputed(Orbital &phi_p) {
    for (int i = 0; i < this->orbitals->size(); i++) {
        Orbital &phi_i  = this->orbitals->getOrbital(i);
        Orbital *ex_i = this->exchange[i];
        if (&phi_i == &phi_p and ex_i != 0) {
            Orbital *result = new Orbital(phi_p);
            // Deep copy of orbital
            OrbitalAdder add(this->apply_prec, this->max_scale);
            add.inPlace(*result, 1.0, *ex_i);
            return result;
        }
    }
    return 0;
}
