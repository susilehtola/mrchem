#include "catch.hpp"

#include "factory_functions.h"
#include "MWProjector.h"
#include "IdentityKernel.h"
#include "CrossCorrelationGenerator.h"
#include "OperatorTree.h"
#include "BandWidth.h"

namespace identity_operator {

TEST_CASE("Initialize identity operator", "[identity_operator], [mw_operator]") {
    double exp_prec  = 1.0e-6;
    double proj_prec = 1.0e-6;
    double ccc_prec  = 1.0e-4;

    SECTION("Initialize identity kernel") {
        IdentityKernel id_kern(exp_prec);
        REQUIRE( id_kern.size() == 1 );

        SECTION("Project identity kernel") {
            MultiResolutionAnalysis<1> *kern_mra = 0;
            initializeKernel(&kern_mra);
            MWProjector<1> Q(*kern_mra, proj_prec);

            FunctionTree<1> *kern_tree = Q(id_kern);
            REQUIRE( kern_tree->integrate() == Approx(1.0).epsilon(proj_prec) );

            SECTION("Build operator tree by cross correlation") {
                MultiResolutionAnalysis<2> *oper_mra = 0;
                initializeOperator(&oper_mra);

                CrossCorrelationGenerator G(*oper_mra, ccc_prec);
                OperatorTree *oper_tree = G(*kern_tree);

                oper_tree->calcBandWidth(1.0);
                BandWidth bw_1 = oper_tree->getBandWidth();
                oper_tree->calcBandWidth(0.001);
                BandWidth bw_2 = oper_tree->getBandWidth();
                oper_tree->calcBandWidth(-1.0);
                BandWidth bw_3 = oper_tree->getBandWidth();

                for (int i = 0; i < oper_tree->getDepth(); i++) {
                    REQUIRE( bw_1.getMaxWidth(i) <= bw_2.getMaxWidth(i) );
                    REQUIRE( bw_2.getMaxWidth(i) <= bw_3.getMaxWidth(i) );
                }

                delete oper_tree;
                finalize(&oper_mra);
            }
            delete kern_tree;
            finalize(&kern_mra);
        }
    }

}

} // namespace
