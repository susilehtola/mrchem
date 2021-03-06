#include <fstream>

#include "Molecule.h"
#include "Nucleus.h"
#include "DipoleMoment.h"
#include "Magnetizability.h"
#include "NMRShielding.h"
#include "HyperFineCoupling.h"
#include "SpinSpinCoupling.h"

using namespace std;
using namespace Eigen;

Molecule::Molecule(const Nuclei &nucs, int c)
        : charge(c),
          nuclei(nucs),
          energy(0),
          dipole(0),
          quadrupole(0),
          magnetizability(0),
          nmr(0),
          hfcc(0),
          sscc(0) {
    calcCenterOfMass();
    allocNuclearProperties();
}

Molecule::Molecule(const string &coord_file, int c)
        : charge(c),
          energy(0),
          dipole(0),
          quadrupole(0),
          magnetizability(0),
          nmr(0),
          hfcc(0),
          sscc(0) {
    readCoordinateFile(coord_file);
    calcCenterOfMass();
    allocNuclearProperties();
}

Molecule::Molecule(const vector<string> &coord_str, int c)
        : charge(c),
          energy(0),
          dipole(0),
          quadrupole(0),
          magnetizability(0),
          nmr(0),
          hfcc(0),
          sscc(0) {
    readCoordinateString(coord_str);
    calcCenterOfMass();
    allocNuclearProperties();
}

void Molecule::allocNuclearProperties() {
    int nNucs = this->nuclei.size();
    this->nmr = new NMRShielding*[nNucs];
    this->hfcc = new HyperFineCoupling*[nNucs];
    this->sscc = new SpinSpinCoupling**[nNucs];
    for (int k = 0; k < nNucs; k++) {
        this->nmr[k] = 0;
        this->hfcc[k] = 0;
        this->sscc[k] = new SpinSpinCoupling*[nNucs];
        for (int l = 0; l < nNucs; l++) {
            this->sscc[k][l] = 0;
        }
    }
}

Molecule::~Molecule() {
    clearSCFEnergy();
    clearDipoleMoment();
    //clearQuadrupoleMoment();
    clearMagnetizability();
    //clearPolarizability();
    //clearOpticalRotation();
    freeNuclearProperties();
    this->nuclei.clear();
}

void Molecule::freeNuclearProperties() {
    int nNucs = this->nuclei.size();
    for (int k = 0; k < nNucs; k++) {
        clearNMRShielding(k);
        clearHyperFineCoupling(k);
        for (int l = 0; l < nNucs; l++) {
            clearSpinSpinCoupling(k, l);
        }
        delete[] this->sscc[k];
        this->sscc[k] = 0; 
    }
    delete[] this->nmr;
    delete[] this->hfcc;
    delete[] this->sscc;
    this->nmr = 0; 
    this->hfcc = 0; 
    this->sscc = 0; 
}

void Molecule::clearSCFEnergy() {
    if (this->energy != 0) {
        delete this->energy;
        this->energy = 0;
    }
}

void Molecule::clearDipoleMoment() {
    if (this->dipole != 0) {
        delete this->dipole;
        this->dipole = 0;
    }
}

void Molecule::clearQuadrupoleMoment() {
    NOT_IMPLEMENTED_ABORT;
    /*
    if (this->quadrupole != 0) {
        delete this->quadrupole;
        this->quadrupole = 0;
    }
    */
}

void Molecule::clearMagnetizability() {
    if (this->magnetizability != 0) {
        delete this->magnetizability;
        this->magnetizability = 0;
    }
}

void Molecule::clearNMRShielding(int k) {
    if (this->nmr == 0) MSG_ERROR("Properties not allocated");
    if (this->nmr[k] != 0) {
        delete this->nmr[k];
        this->nmr[k] = 0;
    }
}

void Molecule::clearHyperFineCoupling(int k) {
    if (this->hfcc == 0) MSG_ERROR("Properties not allocated");
    if (this->hfcc[k] != 0) {
        delete this->hfcc[k];
        this->hfcc[k] = 0;
    }
}

void Molecule::clearSpinSpinCoupling(int k, int l) {
    if (this->sscc == 0) MSG_ERROR("Properties not allocated");
    if (this->sscc[k] == 0) MSG_ERROR("Properties not allocated");
    if (this->sscc[k][l] != 0) {
        delete this->sscc[k][l];
        this->sscc[k][l] = 0;
    }
}

void Molecule::clearPolarizability() {
    NOT_IMPLEMENTED_ABORT;
    /*
    int nPol = this->polarizability.size();
    for (int i = 0; i < nPol; i++) {
        if (this->polarizability[i] != 0) {
            delete this->polarizability[i];
            this->polarizability[i] = 0;
        }
    }
    this->polarizability.clear();
    */
}

void Molecule::clearOpticalRotation() {
    NOT_IMPLEMENTED_ABORT;
    /*
    int nOpt = this->optical_rotation.size();
    for (int i = 0; i < nOpt; i++) {
        if (this->optical_rotation[i] != 0) {
            delete this->optical_rotation[i];
            this->optical_rotation[i] = 0;
        }
    }
    this->optical_rotation.clear();
    */
}

void Molecule::initSCFEnergy() {
    if (this->energy != 0) MSG_WARN("SCFEnergy already initialized");
    this->energy = new SCFEnergy();
}

void Molecule::initDipoleMoment() {
    if (this->dipole != 0) MSG_WARN("Dipole moment already initialized");
    this->dipole = new DipoleMoment();
}

void Molecule::initQuadrupoleMoment() {
    NOT_IMPLEMENTED_ABORT;
    /*
    if (this->quadrupole != 0) {
        MSG_WARN("Quadrupole moment already initialized");
    }
    this->quadrupole = new QuadrupoleMoment();
    */
}

void Molecule::initMagnetizability() {
    if (this->magnetizability != 0) MSG_WARN("Magnetizability already initialized");
    this->magnetizability = new Magnetizability();
}

void Molecule::initNMRShielding(int k) {
    if (this->nmr == 0) MSG_ERROR("Properties not allocated");
    if (this->nmr[k] != 0) MSG_ERROR("NMR shielding tensor already initialized");

    const Nucleus &nuc_K = getNucleus(k);
    this->nmr[k] = new NMRShielding(nuc_K);
}

void Molecule::initHyperFineCoupling(int k) {
    if (this->hfcc == 0) MSG_ERROR("Properties not allocated");
    if (this->hfcc[k] != 0) MSG_ERROR("HyperFine coupling tensor already initialized");

    const Nucleus &nuc_K = getNucleus(k);
    this->hfcc[k] = new HyperFineCoupling(nuc_K);
}

void Molecule::initSpinSpinCoupling(int k, int l) {
    if (this->sscc == 0) MSG_ERROR("Properties not allocated");
    if (this->sscc[k] == 0) MSG_ERROR("Properties not allocated");
    if (this->sscc[k][l] != 0) MSG_ERROR("Spin-spin coupling tensor already initialized");

    const Nucleus &nuc_K = getNucleus(k);
    const Nucleus &nuc_L = getNucleus(l);
    this->sscc[k][l] = new SpinSpinCoupling(nuc_K, nuc_L);
}

void Molecule::initPolarizability(double omega) {
    NOT_IMPLEMENTED_ABORT;
    /*
    for (int i = 0; i < this->polarizability.size(); i++) {
        Polarizability &pol_i = *this->polarizability[i];
        double omega_i = pol_i.getFrequency();
        if (fabs(omega_i - omega) < MachineZero) {
            MSG_ERROR("Polarizability already initialized");
            return;
        }
    }
    Polarizability *pol = new Polarizability(omega);
    this->polarizability.push_back(pol);
    */
}

void Molecule::initOpticalRotation(double omega) {
    NOT_IMPLEMENTED_ABORT;
    /*
    for (int i = 0; i < this->optical_rotation.size(); i++) {
        OpticalRotation &optrot_i = *this->optical_rotation[i];
        double omega_i = optrot_i.getFrequency();
        if (fabs(omega_i - omega) < MachineZero) {
            MSG_ERROR("OpticalRotation already initialized");
            return;
        }
    }
    OpticalRotation *optrot = new OpticalRotation(omega);
    this->optical_rotation.push_back(optrot);
    */
}

SCFEnergy& Molecule::getSCFEnergy() {
    if (this->energy == 0) MSG_ERROR("Uninitialized SCF energy");
    return *this->energy;
}

DipoleMoment& Molecule::getDipoleMoment() {
    if (this->dipole == 0) MSG_ERROR("Uninitialized dipole moment");
    return *this->dipole;
}

QuadrupoleMoment& Molecule::getQuadrupoleMoment() {
    NOT_IMPLEMENTED_ABORT;
    /*
    if (this->quadrupole == 0) MSG_ERROR("Uninitialized quadrupole moment");
    return *this->quadrupole;
    */
}

Magnetizability& Molecule::getMagnetizability() {
    if (this->magnetizability == 0) MSG_ERROR("Uninitialized magnetizability");
    return *this->magnetizability;
}

NMRShielding& Molecule::getNMRShielding(int k) {
    if (this->nmr == 0) MSG_ERROR("Properties not allocated");
    if (this->nmr[k] == 0) MSG_ERROR("Uninitialized NMR shielding tensor " << k);
    return *this->nmr[k];
}

HyperFineCoupling& Molecule::getHyperFineCoupling(int k) {
    if (this->hfcc == 0) MSG_ERROR("Properties not allocated");
    if (this->hfcc[k] == 0) MSG_ERROR("Uninitialized hyperfine coupling tensor " << k);
    return *this->hfcc[k];
}

SpinSpinCoupling& Molecule::getSpinSpinCoupling(int k, int l) {
    if (this->sscc == 0) MSG_ERROR("Properties not allocated");
    if (this->sscc[k] == 0) MSG_ERROR("Properties not allocated");
    if (this->sscc[k][l] == 0) MSG_ERROR("Uninitialized spin-spin coupling tensor " << k << " " << l);
    return *this->sscc[k][l];
}

Polarizability& Molecule::getPolarizability(double omega) {
    NOT_IMPLEMENTED_ABORT;
    /*
    Polarizability *pol_w = 0;
    for (int i = 0; i < this->polarizability.size(); i++) {
        double omega_i = this->polarizability[i]->getFrequency();
        if (fabs(omega_i - omega) < MachineZero) {
            pol_w = this->polarizability[i];
            break;
        }
    }
    if (pol_w == 0) MSG_ERROR("Uninitialized polarizability");
    return *pol_w;
    */
}

OpticalRotation& Molecule::getOpticalRotation(double omega) {
    NOT_IMPLEMENTED_ABORT;
    /*
    OpticalRotation *optrot_w = 0;
    for (int i = 0; i < this->optical_rotation.size(); i++) {
        double omega_i = this->optical_rotation[i]->getFrequency();
        if (fabs(omega_i - omega) < MachineZero) {
            optrot_w = this->optical_rotation[i];
            break;
        }
    }
    if (optrot_w == 0) MSG_ERROR("Uninitialized optical rotation");
    return *optrot_w;
    */
}

int Molecule::getNElectrons() const {
    int totZ = 0;
    for (int i = 0; i < getNNuclei(); i++) {
        totZ += getNucleus(i).getElement().getZ();
    }
    return totZ - this->charge;
}

void Molecule::calcCenterOfMass() {
    this->COM[0] = 0.0;
    this->COM[1] = 0.0;
    this->COM[2] = 0.0;

    double M = 0.0;
    for (int i = 0; i < getNNuclei(); i++) {
        const Nucleus &nuc = getNucleus(i);
        const double *r_i = nuc.getCoord();
        const double m_i = nuc.getElement().getMass();
        for (int d = 0; d < 3; d++) {
            this->COM[d] += r_i[d] * m_i;
        }
        M += m_i;
    }
    for (int d = 0; d < 3; d++) {
        this->COM[d] *= 1.0/M;
    }
}

void Molecule::readCoordinateFile(const string &coord_file) {
    fstream ifs;
    ifs.open(coord_file.c_str());
    if (not ifs) {
        MSG_FATAL("Failed to open basis set file: " << coord_file);
    }

    int nNuclei;
    string sym;
    double coord[3];
    ifs >> nNuclei;
    for (int i = 0; i < nNuclei; i++) {
        ifs >> sym;
        ifs >> coord[0];
        ifs >> coord[1];
        ifs >> coord[2];
        this->nuclei.push_back(sym.c_str(), coord);
    }
    ifs.close();
}

void Molecule::readCoordinateString(const vector<string> &coord_str) {
    int nNuclei = coord_str.size();
    string sym;
    double coord[3];
    for (int i = 0; i < nNuclei; i++) {
        stringstream ss;
        ss.str(coord_str[i]);
        ss >> sym;
        ss >> coord[0];
        ss >> coord[1];
        ss >> coord[2];
        this->nuclei.push_back(sym.c_str(), coord);
    }
}

void Molecule::printGeometry() const {
    TelePrompter::printHeader(0, "Molecule");
    println(0, " Nr  Element             x             y             z      ");
    TelePrompter::printSeparator(0, '-');
    int oldPrec = TelePrompter::setPrecision(5);

    int nNuclei = getNNuclei();
    for (int i = 0; i < nNuclei; i++) {
        const Nucleus &nuc = getNucleus(i);
        const double *coord = nuc.getCoord();
        stringstream symbol;
        symbol << nuc.getElement().getSymbol();
        symbol << "  ";
        printout(0, setw(3) << i+1 << "     ");
        printout(0, symbol.str()[0] << symbol.str()[1]);
        printout(0, setw(21) << coord[0]);
        printout(0, setw(14) << coord[1]);
        printout(0, setw(14) << coord[2] << endl);
    }
    TelePrompter::printSeparator(0, '-');
    printout(0, " Center of mass: ");
    printout(0, setw(14) << this->COM[0]);
    printout(0, setw(14) << this->COM[1]);
    printout(0, setw(14) << this->COM[2] << endl);
    TelePrompter::setPrecision(oldPrec);
    TelePrompter::printSeparator(0, '=', 2);
}

void Molecule::printProperties() const {
    if (this->energy != 0) println(0, *this->energy);
    if (this->dipole != 0) println(0, *this->dipole);
    if (this->magnetizability != 0) println(0, *this->magnetizability);
    if (this->nmr != 0) {
        for (int k = 0; k < this->nuclei.size(); k++) {
            if (this->nmr[k] != 0) println(0, *this->nmr[k]);
        }
    }
    if (this->hfcc != 0) {
        for (int k = 0; k < this->nuclei.size(); k++) {
            if (this->hfcc[k] != 0) println(0, *this->hfcc[k]);
        }
    }
    if (this->sscc != 0) {
        for (int k = 0; k < this->nuclei.size(); k++) {
            for (int l = 0; l < this->nuclei.size(); l++) {
                if (this->sscc[k][l] != 0) println(0, *this->sscc[k][l]);
            }
        }
    }
}
