#include "MWOperator.h"
#include "FunctionTree.h"
#include "WaveletAdaptor.h"
#include "GridGenerator.h"
#include "OperApplicationCalculator.h"
#include "MultiResolutionAnalysis.h"
#include "Timer.h"

template<int D>
FunctionTree<D>* MWOperator<D>::operator()(FunctionTree<D> &inp) {
    FunctionTree<D> *out = new FunctionTree<D>(this->MRA);

    Timer init_t;
    init_t.restart();
    GridGenerator<D> G(this->MRA);
    G(*out, inp);
    init_t.stop();
    println(10, "Time initializing   " << init_t);

    (*this)(*out, inp, -1);
    return out;
}

template<int D>
void MWOperator<D>::operator()(FunctionTree<D> &out,
                               FunctionTree<D> &inp,
                               int maxIter) {
    this->oper.calcBandWidths(this->apply_prec);
    this->adaptor = new WaveletAdaptor<D>(this->apply_prec, this->MRA.getMaxScale());
    this->calculator = new OperApplicationCalculator<D>(this->apply_prec, this->oper, inp);
    this->build(out, maxIter);
    this->clearCalculator();
    this->clearAdaptor();
    this->oper.clearBandWidths();

    Timer post_t;
    post_t.restart();
    out.mwTransform(TopDown, false); // add coarse scale contributions
    out.mwTransform(BottomUp);
    out.calcSquareNorm();
    inp.deleteGenerated();
    post_t.stop();

    println(10, "Time post operator  " << post_t);
    println(10, std::endl);
}

template<int D>
void MWOperator<D>::clearOperator() {
    for (int i = 0; i < this->oper.size(); i++) {
        if (this->oper[i] != 0) delete this->oper[i];
    }
    this->oper.clear();
}

template<int D>
MultiResolutionAnalysis<2>* MWOperator<D>::getOperatorMRA() {
    const BoundingBox<D> &box = this->MRA.getWorldBox();
    const ScalingBasis &basis = this->MRA.getScalingBasis();

    int maxn = 0;
    for (int i = 0; i < D; i++) {
        if (box.size(i) > maxn) {
            maxn = box.size(i);
        }
    }
    int nbox[2] = { maxn, maxn};
    NodeIndex<2> idx(box.getScale());
    BoundingBox<2> oper_box(idx, nbox);
    return new MultiResolutionAnalysis<2>(oper_box, basis);
}

template class MWOperator<1>;
template class MWOperator<2>;
template class MWOperator<3>;