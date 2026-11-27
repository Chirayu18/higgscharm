# Regenerate the running-Yukawa factor tables. Usage: MG=/path/to/mg5amcnlo-v2.9.10-patched bash build.sh
HERE=$(cd "$(dirname "$0")" && pwd)
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null
cd /afs/cern.ch/user/c/cgupta/CMSSW_14_1_0_pre4/src && eval "$(scramv1 runtime -sh)"
cd "$HERE"  # needs MG5_aMC v2.9.10 source with the H+c patch applied at $MG
cat > as2.cc <<'XX'
#include "LHAPDF/LHAPDF.h"
#include <cstdio>
int main(){ LHAPDF::setVerbosity(0); LHAPDF::PDF* p = LHAPDF::mkPDF("NNPDF31_nnlo_as_0118_mc_hessian_pdfas", 0); printf("%.10f\n", p->alphasQ(91.188)); return 0; }
XX
g++ -O1 -o as2 as2.cc $(lhapdf-config --cflags --libs) && AS5=$(LD_LIBRARY_PATH=$(lhapdf-config --libdir):$LD_LIBRARY_PATH ./as2)
echo "alphas(MZ) 325300 = $AS5"
sed -n 53,498p $MG/Template/NLO/SubProcesses/reweight_xsec.f > rundec.f
gfortran -O2 -std=legacy -o tab tab.f rundec.f 2>&1 | grep -v -i warning | grep -i error
echo "0.986 3.0 4.0 0.1126601343" | ./tab > hpluscharm_4fs_table.txt
echo "4.18 0.0 5.0 $AS5" | ./tab > hplusbottom_5fs_table.txt
wc -l hplus*_table.txt; awk 'NR%1500==1' hplusbottom_5fs_table.txt
