#pragma once

#pragma GCC system_header
#include <Eigen/Core>

#include "Nucleus.h"

class SpinSpinCoupling {
public:
    SpinSpinCoupling(const Nucleus &n_k, const Nucleus &n_l)
        : nuc_K(n_k), nuc_L(n_l) {
        this->diamagnetic = Eigen::VectorXd::Zero(3,3);
        this->paramagnetic = Eigen::VectorXd::Zero(3,3);
    }

    virtual ~SpinSpinCoupling() { }

    const Nucleus &getNucleusK() const { return this->nuc_K; }
    const Nucleus &getNucleusL() const { return this->nuc_L; }

    Eigen::MatrixXd get() const { return this->diamagnetic + this->paramagnetic; }
    Eigen::MatrixXd& getDiamagnetic() { return this->diamagnetic; }
    Eigen::MatrixXd& getParamagnetic() { return this->paramagnetic; }

    friend std::ostream& operator<<(std::ostream &o, const SpinSpinCoupling &sscc) {
        Eigen::MatrixXd dia = sscc.diamagnetic;
        Eigen::MatrixXd para = sscc.paramagnetic;
        Eigen::MatrixXd tot = dia + para;

        double isoDShz = dia.trace()/3.0;
        double isoPShz = para.trace()/3.0;
        double isoTShz = isoDShz + isoPShz;

        int oldPrec = TelePrompter::setPrecision(10);
        o<<"                                                            "<<std::endl;
        o<<"================= Spin-Spin Coupling tensor ================"<<std::endl;
        o<<"                                                            "<<std::endl;
        TelePrompter::setPrecision(5);
        o<<std::setw(3)  << sscc.getNucleusK().getElement().getSymbol();
        o<<std::setw(26) << sscc.getNucleusK().getCoord()[0];
        o<<std::setw(15) << sscc.getNucleusK().getCoord()[1];
        o<<std::setw(15) << sscc.getNucleusK().getCoord()[2];
        o<<std::endl;
        o<<std::setw(3)  << sscc.getNucleusL().getElement().getSymbol();
        o<<std::setw(26) << sscc.getNucleusL().getCoord()[0];
        o<<std::setw(15) << sscc.getNucleusL().getCoord()[1];
        o<<std::setw(15) << sscc.getNucleusL().getCoord()[2];
        o<<std::endl;
        o<<"                                                            "<<std::endl;
        TelePrompter::setPrecision(10);
        o<<"                                                            "<<std::endl;
        o<<"-------------------- Isotropic averages --------------------"<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<" Total            (Hz)        " << std::setw(30) << isoTShz  <<std::endl;
        o<<" Diamagnetic      (Hz)        " << std::setw(30) << isoDShz  <<std::endl;
        o<<" Paramagnetic     (Hz)        " << std::setw(30) << isoPShz  <<std::endl;
        o<<"                                                            "<<std::endl;
        o<<"-------------------------- Total ---------------------------"<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<std::setw(19)<<tot(0,0)<<std::setw(20)<<tot(1,0)<<std::setw(20)<<tot(2,0)<<std::endl;
        o<<std::setw(19)<<tot(1,0)<<std::setw(20)<<tot(1,1)<<std::setw(20)<<tot(2,1)<<std::endl;
        o<<std::setw(19)<<tot(2,0)<<std::setw(20)<<tot(1,2)<<std::setw(20)<<tot(2,2)<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<"----------------------- Diamagnetic ------------------------"<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<std::setw(19)<<dia(0,0)<<std::setw(20)<<dia(1,0)<<std::setw(20)<<dia(2,0)<<std::endl;
        o<<std::setw(19)<<dia(1,0)<<std::setw(20)<<dia(1,1)<<std::setw(20)<<dia(2,1)<<std::endl;
        o<<std::setw(19)<<dia(2,0)<<std::setw(20)<<dia(1,2)<<std::setw(20)<<dia(2,2)<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<"----------------------- Paramagnetic -----------------------"<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<std::setw(19)<<para(0,0)<<std::setw(20)<<para(1,0)<<std::setw(20)<<para(2,0)<<std::endl;
        o<<std::setw(19)<<para(1,0)<<std::setw(20)<<para(1,1)<<std::setw(20)<<para(2,1)<<std::endl;
        o<<std::setw(19)<<para(2,0)<<std::setw(20)<<para(1,2)<<std::setw(20)<<para(2,2)<<std::endl;
        o<<"                                                            "<<std::endl;
        o<<"============================================================"<<std::endl;
        o<<"                                                            "<<std::endl;
        TelePrompter::setPrecision(oldPrec);
        return o;
    }
protected:
    const Nucleus nuc_K;
    const Nucleus nuc_L;
    Eigen::MatrixXd diamagnetic;
    Eigen::MatrixXd paramagnetic;
};

