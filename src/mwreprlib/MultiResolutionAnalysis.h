#ifndef MULTIRESOLUTIONANALYSIS_H
#define MULTIRESOLUTIONANALYSIS_H

#include "BoundingBox.h"
#include "ScalingBasis.h"

template<int D>
class MultiResolutionAnalysis {
public:
    MultiResolutionAnalysis(BoundingBox<D> &bb, ScalingBasis &sb)
        : world(bb), basis(sb) { }

    const ScalingBasis &getScalingBasis() const { return this->basis; }
    const BoundingBox<D> &getWorldBox() const { return this->world; }

    bool operator==(MultiResolutionAnalysis<D> &mra) const {
        if (this->basis != mra.basis) return false;
        if (this->world != mra.world) return false;
        return true;
    }
    bool operator!=(MultiResolutionAnalysis<D> &mra) const {
        if (this->basis != mra.basis) return true;
        if (this->world != mra.world) return true;
        return false;
    }

    template<int T>
    friend std::ostream& operator<<(std::ostream &o,
                                    const MultiResolutionAnalysis<T> &mra) {
        o << std::endl;
        o << "***************** MultiResolution Analysis *****************";
        o << std::endl;
        o << std::endl << mra.basis;
        o << std::endl;
        o << std::endl << mra.world;
        o << std::endl;
        o << "************************************************************";
        o << std::endl;
        return o;
    }
protected:
    const ScalingBasis basis;
    const BoundingBox<D> world;
};

#endif // MULTIRESOLUTIONANALYSIS_H