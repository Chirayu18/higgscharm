#!/bin/bash
# Freeze-scan a datacard: nominal, per-nuisance, and AN-comparable groups.
#
#   ./freeze_scan.sh <workspace.root> [mode]
#     mode = single   one-at-a-time freeze of every listed nuisance (default)
#            groups   AN-23-102 Table 17 groups, reporting the 68% band
#            quick    just nominal + autoMCStats + stat-only
#
# Groups match AN-23-102 Table 17 so the two can be compared directly.
# Outputs "<label> <limit>" per line; `groups` mode prints the 16/50/84 band.
set -uo pipefail
WS="${1:?usage: freeze_scan.sh <workspace.root> [single|groups|quick]}"
MODE="${2:-single}"

source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
[ -n "${CMSSW_BASE:-}" ] || { cd "$(dirname "$0")"; }

lim(){ combine -M AsymptoticLimits "$WS" -t -1 --run blind --noFitAsimov \
        --mass 120 $2 -n "_$1" 2>&1 | grep 'Expected 50' | grep -oE '[0-9]+\.[0-9]+'; }
band(){ combine -M AsymptoticLimits "$WS" -t -1 --run blind --noFitAsimov \
        --mass 120 $2 -n "_$1" 2>&1 | grep -E "Expected (16|50|84)" \
        | grep -oE "[0-9]+\.[0-9]+" | paste -sd' '; }

case "$MODE" in
quick)
  echo "nominal            $(lim NOM '')"
  echo "freeze autoMCStats $(lim FMC '--freezeParameters rgx{prop_bin.*}')"
  echo "stat-only          $(lim FALL '--freezeParameters allConstrainedNuisances')"
  ;;
single)
  echo "nominal            $(lim NOM '')"
  for N in scalevar_muF scalevar_muR scalevar_muR_muF \
           xsec_hplusc_4FS_5FS xsec_hplusc_PDF CMS_ctag2d_2022 \
           CMS_scale_j_2022 CMS_res_j_2022 pileup ps_isr ps_fsr \
           alphaS_PDF lumi_13p6TeV xsec_vjets xsec_higgsbkg \
           flavor_composition_ggH BR_HtoWW CMS_negrw_vjets rate_tt; do
    echo "$(printf '%-22s' "$N") $(lim "F_$N" "--freezeParameters $N")"
  done
  echo "autoMCStats (all)      $(lim FMC '--freezeParameters rgx{prop_bin.*}')"
  echo "autoMCStats (SR only)  $(lim FSR '--freezeParameters rgx{prop_binSR_hplusc.*}')"
  echo "stat-only              $(lim FALL '--freezeParameters allConstrainedNuisances')"
  ;;
groups)
  # 16/50/84 so sigma = (84-16)/2 can be formed, the AN's metric
  echo "FULL   $(band FULL '')"
  echo "STAT   $(band STAT '--freezeParameters allConstrainedNuisances')"
  echo "MCSTAT $(band MCSTAT '--freezeParameters rgx{prop_bin.*}')"
  echo "SIGTH  $(band SIGTH '--freezeParameters xsec_hplusc_4FS_5FS,xsec_hplusc_PDF')"
  echo "BKGH   $(band BKGH '--freezeParameters xsec_higgsbkg,flavor_composition_ggH,BR_HtoWW')"
  echo "OTHBKG $(band OTHBKG '--freezeParameters xsec_st,xsec_diboson,xsec_vjets,CMS_negrw_vjets')"
  echo "TTNORM $(band TTNORM '--freezeParameters rate_tt')"
  echo "CTAG   $(band CTAG '--freezeParameters CMS_ctag2d_2022')"
  echo "JESJER $(band JESJER '--freezeParameters CMS_scale_j_2022,CMS_res_j_2022')"
  echo "LEPTON $(band LEPTON '--freezeParameters muon_id,muon_iso,electron_id,electron_reco_RecoBelow20,electron_reco_Reco20to75,electron_reco_RecoAbove75,CMS_scale_e_2022,CMS_res_e_2022,CMS_scale_m_2022,CMS_res_m_2022')"
  echo "PILEUP $(band PILEUP '--freezeParameters pileup')"
  echo "THEORY $(band THEORY '--freezeParameters scalevar_muR,scalevar_muF,scalevar_muR_muF,ps_isr,ps_fsr,alphaS_PDF')"
  ;;
*) echo "unknown mode: $MODE"; exit 2 ;;
esac
