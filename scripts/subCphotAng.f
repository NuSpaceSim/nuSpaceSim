        subroutine CphotAngElementwise(betaE,alt,dphots,cang)
c  20-Nov-19: put in condition that betaE cannot be lower than 1 deg
c   try to solve the python crash with error "Bus error: 10"
c
c  19-Nov-19: per Alex's suggestion, do not pass the output in a common block
c       since this (in principle) can be overwritten by some other routine
c       best to pass the return values as args in the subroutine
c       this uses Cf2py definition of the intent (input vs output) of the variables
c
c
c  19-Nov-19: removed print statement that reported dphots,cang
c      Note that dphots calculated photons/m^2 at 525 km altitude for 100 PeV EASs
c       so need to scale by energy of EAS (E_EAS/100 PeV) 
c        to get dphots for an EAS of energy E_EAS
c
c  15-Sep-19: changed atan to atan2 to solve gfortran-mp-9 optimization issue
c
c
c   28-Aug-19: replace eang (real) loop with integer loop
c     to remove warnings from gfortran compilers
c   
c      also cleanup up code to remove warnings regarding unused variables/labels
c      also removed double presision definition of AirN (check shows it doesn't matter)
c      also replace kappa bt Okappa (since should be real)
c
c   20-Aug-19: Code to be put in master loop
c     to provide PhotDensity and Cang same
c     as done via transfer.f done for original calc for POEMMA
c     PhysRevD and ToO paper
c
c     Assumptions: 100 PeV EAS for now
c
c     Input: Beta (Earth-emergence Angle in deg)
c            Altitude (km where to shart EAS)
c     
c     Output: PhotDensity (photons/m^2 in Cherenkov ring)
c             Cherenkov Angle (deg)
c           
c     Cleaned up code to remove things not needed
c

c
c   8-May-18
c     corrected OD interprolation to
c          tmpOD=aOD55(int(z)+1)-(z-real(int(z)))*dfaOD55(int(z)+1)
c      was
c          tmpOD=aOD55(int(z)+1)-z*dfaOD55(int(z)+1)
c
c
c   13-Apr-14: fixed distance to instrument in histogram
c       projection ... was using zmax/cos(ThetView)
c


        parameter (nWave=29)
        dimension parin(9),photsum(1),showin(4)
        dimension cpick(3000),ctpick(3000)
c       dimension ctbin(6000),ctwth(6000)
 
        dimension OzDepth(11), OzDsum(11), OzZeta(11) 
        dimension wave1(nWave),clight(3000,nWave)
        dimension Pyield(nWave)
        dimension SPyield(nWave)
        dimension wmean(nWave), TrRayl(nWave),cttot(3000)
        dimension aOD55(30),dfaOD55(30)
        dimension aODwave(nWave),aBeta(nWave),aBetaF(nWave)
        parameter (NSTEP=35000)
c taken out 20-Mar-18 should go to NSTEP
c        dimension ctrck1(360),ctrck2(360),ctrck3(360)

        dimension delgram(NSTEP),ZonZ(NSTEP)
        dimension gramz(NSTEP),gramsum(NSTEP),zsave(NSTEP)
        dimension ThetPrpA(NSTEP)
        dimension taphotstep(NSTEP)
        dimension CangStep(NSTEP),DistStep(NSTEP)


        real betaE,Alt
        real Dphots,Cang
Cf2py   intent(in) betaE
Cf2py   intent(in) Alt
Cf2py   intent(out) Dphots
Cf2py   intent(out) Cang

c 19-Nov-19 removed common since passing output as subroutine args
c        common /EASdata/  Dphots,Cang
c 28-aug-19 removed this to address gfortran warnings
c        double precision AirN

      data wave1/200., 225., 250., 275., 
     + 300., 325., 350., 375., 
     + 400., 425., 450., 475., 
     + 500., 525., 550., 575.,
     + 600., 625., 650., 675.,
     + 700., 725., 750., 775.,
     + 800., 825., 850., 875.,900./

       data aODwave/0.458,0.458,0.458,0.427,
     + 0.411,0.395,0.379,0.364,
     + 0.316,0.316,0.364,0.364,
     + 0.264,0.264,0.250,0.250,
     + 0.237,0.237,0.224,0.224,
     + 0.213,0.213,0.213,0.213,
     + 0.201,0.201,0.201,0.201,0.19/

      data aBeta/0.29,0.29,0.29,0.27,
     + 0.26,0.25,0.23,0.23,
     + 0.2,0.2,0.18,0.18,
     + 0.167,0.167,0.158,0.158,
     + 0.15,0.15,0.142,0.142,
     + 0.135,0.135,0.135,0.135,
     + 0.127,0.127,0.127,0.127,0.12/


       data OzZeta/5.35, 10.2, 14.75, 19.15, 23.55, 
     + 28.1, 32.8, 37.7, 42.85, 48.25, 100./

       data OzDepth/ 15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 
     + 57.0, 29.4, 10.9, 3.2, 1.3/

      data OzDsum/ 310., 301., 291., 260., 189., 101.8, 
     + 44.8, 15.4, 4.5, 1.3, 0.1/

      data aOD55/0.250,0.136,0.086,0.065,0.055,0.049,0.045,
     + 0.042,0.038,0.035,0.032,0.029,0.026,0.023,0.020,0.017,
     + 0.015,0.012,0.010,0.007,0.006,0.004,0.003,0.003,0.002,
     + 0.002,0.001,0.001,0.001,0.001/



c    Inputs:  height (z), step size (dL)
c      Assumptions:  1000 km Orbit
c
c    parin(1)   step size (dL) km      
c    parin(2)   orbit height km
c    parin(3)   bin size of histogram (km)
c    parin(4)   if = 0 integrate Cherenkov over all wavelengths
c               if <> 0, = wavelength bin number
c    parin(5)  record time dispersion at this radial point (km)
c    parin(6)  +/- max value on histogram (km)
c    parin(7)  Earth emerge theta of shower (degrees)
c    parin(8)  log(E) E in eV
c    parin(9)  starting alt of shower

       parin(1)=0.1
       parin(2)=525.
c       parin(2)=36.
c       parin(2)=33.
c       parin(2)=4.
       parin(3)=4. ! 525 km
c       parin(3)=2. ! 33 km, 525 km higher alt
c       parin(3)=0.1 ! 2 km
c       parin(3)=0.25 ! 4 km
       parin(4)=0.
       parin(5)=.5
       parin(6)=100. ! 525 km
c       parin(6)=1000. 
c       parin(6)=48. ! 525 km high alt
c       parin(6)=24. ! 525 km higher alt
c       parin(6)=20. ! 33 km
c       parin(6)=2. ! 2 km
c       parin(6)=10. ! 4 km
c       parin(7)=10.
c        parin(7)=1.
c       parin(7)=5.
c        parin(7)=35.

c       print*,' enter Beta_e (deg)'
c       read*,betaE

c      betaE is subroutine input
c 20-Nov-19, if betaE < 1 deg, set to 1 deg for calc
c according to Toni this is what she did for the POEMMA
c calc based on my EAS data since 1 deg was lowest value

c        print*,' betaE = ',betaE ! debug use
        if (betaE.lt.1.) then
         betaE=1.
c         print*,' betaE set to 1'
        endif
        parin(7)=betaE


c       parin(7)=1.
c       parin(7)=49.
c       parin(8)=17.78 ! 600 PeV
c       parin(8)=17.875 ! 750 PeV
c        parin(8)=16.7 ! 50 PeV
c        parin(8)=17.6 ! 400 PeV
c        parin(8)=16.48 ! 30 PeV
c        parin(8)=19.48 ! 30 EeV
c       parin(8)=16.3 ! 20 PeV
c       parin(8)=17.3 ! 200 PeV
c       parin(8)=19.
        parin(8)=17.
c       parin(9)=0. ! starting alt
        parin(9)=20.

       photsum(1)=0.
       AltMax=parin(2)

       nbin=int(parin(6)/parin(3))*2


       aBeta55=0.158
c  parameters for 1/Beta fit vs wavelength
c     5th order polynomial

       aP0=-1.2971
       aP1=0.22046E-01
       aP2=-0.19505E-04
       aP3=0.94394E-08
       aP4=-0.21938E-11
       aP5=0.19390E-15

       do k=1,nWave-1
        wmean(k)=wave1(k)+12.5
        tBetinv=aP0+aP1*wmean(k)+aP2*wmean(k)**2+aP3*wmean(k)**3+
     +   aP4*wmean(k)**4+aP5*wmean(k)**5
        aBetaF(k)=1./tBetinv
       enddo
c
c calc OD/km difference
c
       do i=1,29
        dfaOD55(i)=aOD55(i)-aOD55(i+1)
       enddo
        dfaOD55(30)=0.

       Pi=3.1415926
       Alpha=1./137.04

c       do 888 iAlt=0,20
c       print*,' enter iAltitude (km)'
c       read*,Alt

c       Alt is subroutine input

       photsum(1)=0.

       do k=1,NSTEP
        taphotstep(k)=0.
        CangStep(k)=0.
        DistStep(k)=0.
       enddo

       dL=parin(1)
       if (dL.lt.0.01) then
        dL=0.01
        print*,' dL to small, set to 0.01 km'
       endif
       zmax=parin(2)

      xold=1032.9414
      RadE=6378.14
      Rad=RadE

       ThetProp=parin(7)*Pi/180.

       ThetView=asin(RadE/(RadE+zmax)*cos(ThetProp))


       if (dcirc.lt.0.01) then
        print*,' Dcirc to small, set = 0.01'
        dcirc=0.01
        parin(3)=0.01
       endif

       taphotsum=0.
c
c   calc atmosphere at zmax
c
c     Calculate Grammage
       if (zmax.lt.11.) then
        a1=44.34
        a2=-11.861
        a3=1./0.19
        a4=a3-1.
        Xmax=((zmax-a1)/a2)**a3
        rhomax=-1.e-5*a3/a2*((zmax-a1)/a2)**a4
       else if (zmax.lt.25.) then
        a1=45.5
        a2=-6.34
        Xmax=exp((zmax-a1)/a2)
        rhomax=-1.e-5/a2*exp((zmax-a1)/a2)
       else
        a1=13.841
        a2=28.920
        a3=3.344
        Xmax=exp(a1-sqrt(a2+a3*zmax))
        rhomax=0.5e-5*a3/sqrt(a2+a3*zmax)*
     +  exp(a1-sqrt(a2+a3*zmax))
       endif
c   calc ozone depth

      do i=10,1,-1
        if (zmax.gt.OzZeta(i)) then
             sum1=OzDsum(i+1)
             index=i
             goto 15
         endif
      enddo

       index=1

15      if (zmax.lt.5.35) then
           Zonmax=310.+((5.35-zmax)/5.35)*15.
        elseif (zmax.gt.100.) then
            Zonmax=0.1
        else
           Zonmax=sum1+((OzZeta(index+1)-zmax)/
     +     (OzZeta(index+1)-OzZeta(index)))*
     +     Ozdepth(index+1)
        endif
c
c   Determine Rayleigh and Ozone slant depth
c
      iz=0
c      Zold=0.
c      Zold=parin(9) ! this now is set in loop
c      Zold=real(iAlt)
      Zold=Alt
      z=Zold

c        Calculate Ozone Losses

        do i=10,1,-1
         if (z.gt.OzZeta(i)) then
             sum1=OzDsum(i+1)
             index=i
             goto 17 
         endif
        enddo

        index=1
17      if (z.lt.5.35) then
           TotZon=310.+((5.35-z)/5.35)*15.
        elseif (z.gt.100.) then
            TotZon=0.1
        else
           TotZon=sum1+((OzZeta(index+1)-z)/
     +     (OzZeta(index+1)-OzZeta(index)))*
     +     Ozdepth(index+1)
        endif

        ZonBegin=TotZon

        ZonOld=TotZon
        GramRayl=0.
        GramTsum=0.
        ZoneSum=0.

c      Zold=0.
c        Zold=parin(9)
c        Zold=real(iAlt)
       Zold=Alt
        Rad=RadE+Zold

        zMaxZ=65. ! do dL prop until this altitude

c      do xL=dL,35.,dL
18     continue
       iz=iz+1

c
c  correct ThetProp for starting altitude
c

        ThetProp=acos((RadE+zmax)/(RadE+Zold)*sin(ThetView))



       delz=sqrt(Rad**2+dL**2-2.*Rad*dL*cos(Pi/2.+ThetProp))-
     +  Rad

c   set z to measure rho at center of bin
c 20-Mar-18 need to think about this ...
       z=Zold+delz/2.
       Zold=z
   
       zsave(iz)=z

c   add in the other 1/2 to record top of bin
       Zold=z+delz/2.

c     Calculate Grammage
       if (z.lt.11.) then
        a1=44.34
        a2=-11.861
        a3=1./0.19
        a4=a3-1.
        X=((z-a1)/a2)**a3
         rho=-1.e-5*a3/a2*((z-a1)/a2)**a4
       else if (z.lt.25.) then
        a1=45.5
        a2=-6.34
        X=exp((z-a1)/a2)
        rho=-1.e-5/a2*exp((z-a1)/a2)
       else
        a1=13.841
        a2=28.920
        a3=3.344
        X=exp(a1-sqrt(a2+a3*z))
        rho=0.5e-5*a3/sqrt(a2+a3*z)*
     +  exp(a1-sqrt(a2+a3*z))
       endif

       delgram(iz)=rho*dL*1.e5
       GramTsum=GramTsum+delgram(iz)
       gramsum(iz)=GramTsum
       gramz(iz)=X

c
c    add in to previous 
c
       do j=1,iz-1
        delgram(j)=delgram(j)+delgram(iz)
       enddo

       GramRayl=delgram(iz)+GramRayl

c   Calc Ozone slant depth

c        Calculate Ozone Losses

c    set z to top of bin

        z=Zold

        do i=10,1,-1
         if (z.gt.OzZeta(i)) then
             sum1=OzDsum(i+1)
             index=i
             goto 19
         endif
        enddo

        index=1
19      if (z.lt.5.35) then
           TotZon=310.+((5.35-z)/5.35)*15.
        elseif (z.gt.100.) then
            TotZon=0.1
        else
           TotZon=sum1+((OzZeta(index+1)-z)/
     +     (OzZeta(index+1)-OzZeta(index)))*
     +     Ozdepth(index+1)
        endif

       ZonZ(iz)=(ZonOld-TotZon)/delz*dL

c
c    add in to previous 
c
        do j=1,iz-1
         ZonZ(j)=ZonZ(j)+ZonZ(iz)
        enddo

        ZoneSum=ZonZ(iz)+ZoneSum
        ZonOld=TotZon

        Radold=Rad
        Rad=Rad+delz
        izmax=iz

        ThetProp=acos((RadE+zmax)/Rad*sin(ThetView))
        ThetPrpA(iz)=ThetProp
        
c use this to not do real do loop
        if (Zold.le.zMaxZ) goto 18


c jump over printing interesting things
       goto 78

c
c      Check summation vs esitmate of grammage and Ozone depth
c


78      continue
c
c    Initialize Variables
c
       xold=1032.9414
       RadE=6378.14
       Rad=RadE

       iz=0
c       Zold=parin(9)
c        Zold=real(iAlt)
       Zold=Alt

       RN=0.
       T=0.
       limit=0
       itind=0

       RNmax=0.
       izRNmax=0

c
c Greissen variables
c
        Eshow=10.**(parin(8)-9.)
c        EshowMeV=10.**(parin(8)-6.)
        Zair = 7.4 !air
        ecrit = 0.710/(Zair+0.96) ! air, GEV (PDG 1996)
        x0=36.66 ! Rad length Air

        beta=log(Eshow/ecrit)

       istart=0

       AveCangI=0.

       do 85 iz=1,izmax

       z=zsave(iz)
       if (z.gt.AltMax) then
        print*,' z > AltMax'
        goto 99
       endif

c     Calculate Index of Refraction

       X=gramz(iz)

       if (X.ne.0) then
        AirN=1.0+(0.000296)*(X/1032.9414)*(273.2/(204.+0.091*X))
       else
        AirN=0.
       endif

       if ((AirN.eq.1).or.(AirN.eq.0)) then
        print*,' AirN = 1 or 0 '
        goto 99
       endif

c
c  Calc Cherenkov Threshold
c
       if ((AirN.ne.0).and.(AirN.ne.1)) then
        eCthres=0.511/sqrt(1.-1./AirN**2)
       else
c  added this 28-aug-19 to remove gfortran compiler warning
c      just need to set to a large result
        eCthres=1.e6
       endif

c
c      Calculate Cerenkov Angle
c
       if (AirN.ne.0) then
        thetaC=acos(1./AirN)  
       else
        thetaC=0.
       endif

       Theta=thetaC
       CangStep(iz)=Theta

c
c    do greissen param

        t=gramsum(iz)/x0
        s=3.*t/(t+2.*beta)

        RN=0.31/sqrt(beta)*exp(t*(1.-3./2.*log(s)))

c debug
c        print*,z,x,AirN,thetaC,s,RN

        if (RN.lt.0) RN=0.
c        if ((RN.lt.10.).and.(s.gt.1)) goto 99
        if ((RN.lt.1.).and.(s.gt.1)) then
c         print*,' Shower died after s > 1'
         goto 99
        endif

        if (RN.gt.RNmax) then
         RNmax=RN
         izRNmax=iz
        endif


        if (s.ge.0.4) then
         E0=44.-17.*(s-1.46)**2
        else
         E0=26.
        endif

         Tfrac=((0.89*E0-1.2)/(E0+eCthres))**s/
     +    (1.+1.e-4*s*eCthres)**2


c  4-Apr-18 moved this before wavelength loop
c   had it latter and it was doing the 1st
c   wmean calc before kicking out
c
c  Note hillas says for s < 0.2
c      e2hill ~ log(s)
c
        e2hill=(1150.+454.*log(s))
c
c     Jump over if e2hill < 0 as
c      shower to young (in s)
c      to use this parameterization
c
       if (e2hill.lt.0) goto 85

c
c    Determine geometry
c

       AngE=Pi/2.-ThetView-ThetPrpA(iz)
       DtoDet=sin(AngE)/sin(ThetView)*(RadE+z)
       if (istart.eq.0) then
c        print*,' DtoDet ',DtoDet
        istart=1
       endif

       DistStep(iz)=DtoDet

       do k=1,nWave-1

c      Calculate Light Yield
       
       PYield(k)=2.*Pi*Alpha*(sin(thetaC))**2*
     +   (1./wave1(k)-1./wave1(k+1))*1.e9

       SPYield(k)=PYield(k)*dL*1000.*RN


c      Calculate Losses due to Rayleigh Scattering


c
c    Calc Rayleigh loss
c

        TrRayl(k)=exp(-1.*delgram(iz)/2974.*
     +   (400./wmean(k))**4)


         SPYield(k)=SPYield(k)*TrRayl(k)

c        Calculate Ozone Losses
c          Ozone atten parameter given by R. McPeters
c
c            Ozone Trans = exp(-kappa dx)
c               where dx=ozone slant depth in atm-cm
c               and kappa = 110.5 x wave^(-44.21) in atm-cm^-1

        Okappa=10.**(110.5-44.21*log10(wmean(k)))
        TrOz=exp(-1.e-3*ZonZ(iz)*Okappa)

        SPYield(k)=SPYield(k)*TrOz

c put in aerosol model based on 550 nm
c     Elterman results
c
        if (z.lt.30) then 
c ver 2 replace aBeta with fit values aBetaF
c         aODepth=aOD55(int(z+1.))*aBeta(k)/aBeta55
c         aODepth=aOD55(int(z+1.))*aBetaF(k)/aBeta55
c         tmpOD=aOD55(int(z)+1)-z*dfaOD55(int(z)+1) ! Ver 3
         tmpOD=aOD55(int(z)+1)-(z-real(int(z)))*dfaOD55(int(z)+1)
c 
c print values for iz=1
c
         if (istart.eq.0) then 
c          print*,'z,aOD55(int(z)+1),dfaOD55(int(z)+1),tmpOD ',
c     +     z,aOD55(int(z)+1),dfaOD55(int(z)+1),tmpOD
cc          istart=1
         endif

         aODepth=tmpOD*aBetaF(k)/aBeta55
         aTrans=exp(-1.*aODepth/cos(Pi/2.-ThetPrpA(iz)))
c         if (int(z).eq.0) print*,' wave,aODept,aTrans ',
c     +    wave1(k),aODepth,atrans
c         if (int(z).eq.0) 
c     +    print*,' z,wmean,aOD55(int(z)+1),aODept,aTrans ',
c     +    z,wmean(k),aOD55(int(z)+1),aODepth,atrans
        else
         aTrans=1.
        endif

c        print*,' b4 mie',wmean(k),SPYield(k)        
        SPYield(k)=SPYield(k)*aTrans
c        print*,' after mie',wmean(k),SPYield(k)

        taphotsum=taphotsum+SPYield(k)*Tfrac
        taphotstep(iz)=taphotstep(iz)+SPYield(k)*Tfrac

        AveCangI=AveCangI+SPYield(k)*Tfrac*thetaC

c   no dphi loops, ignore (100)^2 correction

       sigval=SPYield(k)/((parin(3)*1.e3)**2)


c
c   set limits by distance to det
c     and Cherenkov Angle
c

       CradLim=DistStep(iz)*tan(CangStep(iz))
       dcirc=1.
       jlim=int(CradLim/dcirc)

       do j=1,jlim

        rprime=(REAL(j)-0.5)*(dcirc)

        thetj1=atan2(real(j)*dcirc,DtoDet)
        thetj2=atan2(real(j-1)*dcirc,DtoDet)

        thetaj=atan2(rprime,DtoDet)
c
c
c     Calc ang spread ala Hillas
c

       do Ieang=1,int(log10(Eshow))-2+3

        eang=real(Ieang)

        ehill=10.**eang
        ehill2=10.**(eang+1.)

        ehillave=5.*10.**eang


c
c   Calc effective tracklength for particles
c    as a function of shower age and Cherenkov Threshold
c

        if (s.ge.0.4) then
         E0=44.-17.*(s-1.46)**2
        else
         E0=26.
        endif

        if (ehill.lt.eCthres) then
         tracklen1=((0.89*E0-1.2)/(E0+eCthres))**s/
     +    (1.+1.e-4*s*eCthres)**2
         ehillave=(eCthres+ehill2)/2.
        else
         tracklen1=((0.89*E0-1.2)/(E0+ehill))**s/
     +    (1.+1.e-4*s*ehill)**2
        endif

        if (ehill2.lt.eCthres) then
         tracklen2=((0.89*E0-1.2)/(E0+eCthres))**s/
     +    (1.+1.e-4*s*eCthres)**2
        else
         tracklen2=((0.89*E0-1.2)/(E0+ehill2))**s/
     +    (1.+1.e-4*s*ehill2)**2
        endif

        deltrack=tracklen1 - tracklen2


        if (deltrack.lt.0.) deltrack=0.


        whill=2.*(1.-cos(thetaj))*(ehillave/21.)**2

c
c  location prior to 4-Apr-18
c
c  Note hillas says for s < 0.2
c      e2hill ~ log(s)
c
c        e2hill=(1150.+454.*log(s))
c
c     Jump over if e2hill < 0 as
c      shower to young (in s)
c      to use this parameterization
c
c       if (e2hill.lt.0) goto 85

        vhill=ehillave/e2hill

        wave=0.0054*ehillave*(1.+vhill)/
     +   (1.+13.*vhill+8.3*vhill**2)

        uhill=whill/wave
c
c   Note the following are for low
c    energy showers
c
        z0hill=0.59
        ahill=0.777
        if (sqrt(uhill).lt.z0hill) then
         a2hill=0.478
        else
         a2hill=0.380
        endif


        if ((AirN.ne.0).or.(AirN.ne.1)) then

        uhi=2.*(1.-cos(thetj1))*(ehillave/21.)**2/wave
        ulow=2.*(1.-cos(thetj2))*(ehillave/21.)**2/wave
        ubin=uhi-ulow
        if (ubin.lt.0) ubin=0.

        sigval2=
     +   ahill*exp(-1.*(sqrt(uhill)-z0hill)/a2hill)*
     +   deltrack*sigval*ubin
        if (sigval2.gt.0) then 
c
c  comment 23-Jul-18: the following should have been
c    under the phic loops, thus it is low by 1.e4
c    since took this out earlier
c 

         photsum(1)=photsum(1)+
     +    sigval2*(parin(3)*1.e3)**2

         endif
        endif
c end wavelenth loop
       enddo



       enddo


       enddo


c  next lines print proper shower step params



85     continue


99     continue
        
       goto 444

444    continue

       AveCangI=AveCangI/taphotsum

       nAcnt=0
       CangsigI=0.

       do i=1,NSTEP
        if ((taphotstep(i)*CangStep(i)).ne.0) then
         nAcnt=nAcnt+1
         CangsigI=CangsigI+
     +    taphotstep(i)/taphotsum*(CangStep(i)- AveCangI)**2
        endif
       enddo

        CangsigI=sqrt(CangsigI*real(nAcnt)/real(nAcnt-1))

       goto 555

555    continue

c     this appears to reproduce the 3D profile spread better
       CangArea=AveCangI
       CherArea=pi*(DistStep(izRNmax)*tan(CangArea)*1.e3)**2
       photonDen=0.5*photsum(1)/CherArea
        Dphots=photonDen
        Cang=(AveCangI+CangsigI)*180./pi

c 888    continue

c 999    continue

       end

       

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        subroutine CphotAng(betaE, alt, dphots, cang, N)
C
C       Vectorized call of CphotAngElementwise
C

        implicit none

        INTEGER  N, i
        real betaE(N), Alt(N)
        real Dphots(N), Cang(N)
        real a, b, c, d
Cf2py   intent(in) betaE
Cf2py   intent(in) Alt
Cf2py   intent(out) Dphots
Cf2py   intent(out) Cang
Cf2py integer intent(hide),depend(betaE) :: N=shape(betaE,0)

        DO i = 1,N
            b = betaE(i)
            a = alt(i)
            call CphotAngElementwise(b, a, d, c)
            dphots(i) = d
            cang(i) = c
        ENDDO

        END



