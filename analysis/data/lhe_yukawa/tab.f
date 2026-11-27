      program tab
c  Exact re-evaluation of the patched rwgt_muR_dep_fac used in the central
c  HPlusCharm-4FS / HPlusBottom_5FS MuRFScaleDynX0p50 FxFx gridpacks
c  (MG5_aMC 2.9.9 + <sample>_mg5amc.patch). Routines runalpha/runmass are
c  compiled from MG5_aMC v2.9.10 Template/NLO/SubProcesses/reweight_xsec.f.
c  stdin: yq(param card)  refscale(0 = use mbmb itself)  nf  alphas(MZ)
c  stdout: mu0  F(mu0,mu0)  F(0.5mu0,mu0)  F(2mu0,mu0)   with F=(m(muR)/mbmb)^2
      implicit none
      double precision pi,mz,aewm1,gf,yq,mw,sw,ee,vevmg,mdly,vev,mbmb
      double precision apimz,mu0,f(3),k(3),scale,central,ref,nf,asmz
      double precision apimur,apimb,mbmur,apicentral,mbcentral
      integer i,j,n
      parameter (pi=3.14159265358979323846d0)
      read(*,*) yq,ref,nf,asmz
      mz=91.188d0
      aewm1=132.507d0
      gf=1.16639d-5
      mw=dsqrt(mz**2/2d0+dsqrt(mz**4/4d0
     &   -(pi/aewm1)*mz**2/(gf*dsqrt(2d0))))
      sw=dsqrt(1d0-mw**2/mz**2)
      ee=2d0*dsqrt(pi/aewm1)
      vevmg=2d0*mw*sw/ee
      mdly=yq*dsqrt(2d0)/vevmg
      vev=246.21845813429518469305d0
      mbmb=mdly*vev/dsqrt(2d0)
      if (ref.le.0d0) ref=mbmb
      apimz=asmz/pi
      write(0,*) 'mbmb ',mbmb,' ref ',ref,' nf ',nf,' asMZ ',asmz
      k(1)=1d0
      k(2)=0.5d0
      k(3)=2d0
      n=6000
      do i=0,n
         mu0=10d0**(dlog10(1.0d0)
     &      +i*(dlog10(14000d0)-dlog10(1.0d0))/n)
         central=mu0
         do j=1,3
            scale=k(j)*mu0
            if(dabs(scale/central-1d0).lt.1d-9) then
               CALL runalpha(apimZ,mz,central,nf,4,0,apimuR)
               CALL runalpha(apimZ,mz,ref,nf,4,0,apimb)
               CALL runmass(mbmb,apimb,apimuR,nf,4,mbmuR)
            else
               CALL runalpha(apimZ,mz,central,nf,4,0,apicentral)
               CALL runalpha(apimZ,mz,ref,nf,4,0,apimb)
               CALL runmass(mbmb,apimb,apicentral,nf,4,mbcentral)
               CALL runalpha(apicentral,central,scale,nf,2,0,apimuR)
               CALL runmass(mbcentral,apicentral,apimuR,nf,2,mbmuR)
            endif
            f(j)=(mbmuR/mbmb)**2
         enddo
         write(*,'(4es24.15)') mu0,f(1),f(2),f(3)
      enddo
      end
