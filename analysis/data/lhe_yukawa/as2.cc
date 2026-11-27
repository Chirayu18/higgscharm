#include "LHAPDF/LHAPDF.h"
#include <cstdio>
int main(){ LHAPDF::setVerbosity(0); LHAPDF::PDF* p = LHAPDF::mkPDF("NNPDF31_nnlo_as_0118_mc_hessian_pdfas", 0); printf("%.10f\n", p->alphasQ(91.188)); return 0; }
