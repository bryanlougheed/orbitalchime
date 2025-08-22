===========================                         ============================
                         ASTRONOMIE ET SYSTEMES DYNAMIQUES
                         
                         INSTITUT DE MECANIQUE CELESTE

                                    La93

                                2004, May 27 
===========================                         ============================



The program insola is derived from the same program from
the Earth paleoclimate solution La93 (Laskar, Joutel, Boudin, 1993) with 
several differences.

The utilisation has been greatly simplified, as everything is now 
in a single application, insola, that allows to compute all 
insolation quantities for any time interval between the limits 
of the input files defined in insola.par (nomclimapos, nomclimaneg).

Once the application is generated, the only file to edit is 
insola.par.


We have provided solution La93 from -20 to + 10 Myr.




La93    : 
==========

Main reference :
  La93: Laskar, J., Joutel, F., Boudin, F.: 1993, Orbital, precessional
                  and insolation quantities for the Earth
                  from -20 Myr to + 10Myr
                  Astron. Astrophys. 270, 522


********************************************************************************
*  Authors: J. Laskar, M. Gastineau, F. Joutel                                 *
*  (c) Astronomie et Systemes Dynamiques, Institut de Mecanique Celeste,       *
*      Paris (2004)                                                            *
*                                                                              *
*                               Jacques Laskar                                 *
*                               Astronomie et Systemes Dynamiques,             *
*                               Institut de Mecanique Celeste                  *
*                               77 av. Denfert-Rochereau                       *
*                               75014 Paris                                    *
*                       email:  laskar@imcce.fr                                *
*                                                                              *
********************************************************************************




CONTAINS:
=========



File Summary:
--------------------------------------------------------------------------------
 FileName                 Explanations
--------------------------------------------------------------------------------
README                    This file
INSOLN.LA93_01.BTL.ASC    Nominal solution La93(0,1),
                                    before present years (-20Myr to 0)
INSOLP.LA93_01.BTL.ASC    Nominal solution La93(0,1)
                                    after present years (0 to +10Myr)
INSOLN.LA93_11.BTL.ASC    Nominal solution La93(1,1),
                                    before present years (-20Myr to 0)
INSOLP.LA93_11.BTL.ASC    Nominal solution La93(1,1)
                                    after present years (0 to +10Myr)
Makefile                  to compile and generate derived files
insola.par                Parameter file for insola.
insola.f                  FORTRAN program for computations of various
                                    insolation quantities
insolsub.f                subroutines for insola.f
prepsub.f                 FORTRAN subroutine for insola.f
prepinsol.f               FORTRAN subroutine for preparation step for insola.f
--------------------------------------------------------------------------------


Byte-per-byte Description of file: INSOLN.LA93_01.BTL.ASC
Byte-per-byte Description of file: INSOLP.LA93_01.BTL.ASC
Byte-per-byte Description of file: INSOLN.LA93_11.BTL.ASC
Byte-per-byte Description of file: INSOLP.LA93_11.BTL.ASC
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1-14   F8.3   1000yr  t        Time from J2000  in 1000 years
  18-39   D23.16 ---     e        eccentricity
  43-64   D23.16 rad     eps      obliquity (radians)
  68-89   D23.16 rad     pibar    longitude of perihelion from moving equinox
                                  (radians)
--------------------------------------------------------------------------------


Description:

  La93 is a program for computing the precession and obliquity of
  the Earth for various values of
  1) the tidal effect of the Moon (CMAR)
  2) the dynamical ellipticity of the Earth (FGAM)

  The general solution will be called La93(CMAR,FGAM), thus
  La90 = La93(0.,1.)
    and  La93(1.,1.) is obtained with the same tidal effect as
  in Quinn, Tremaine, Duncan (1991), although these solutions
  are not completely identical (see Laskar, Joutel, Boudin, 1993)
  


BIBLIOGRAPHY:
============



  La90: Laskar, J.: 1990, The chaotic motion of the solar system.
                  A numerical estimate of the chaotic zones
                  Icarus, 88, 266

  La93: Laskar, J., Joutel, F., Boudin, F.: 1993, Orbital, precessional
                  and insolation quantities for the Earth
                  from -20 Myr to + 10Myr
                  Astron. Astrophys. 270, 522

  La2004: Laskar, J., Gastineau, M., Joutel, F., Robutel, P., Levrard, B., 
                  Correia, A.,
                  A long term numerical solution for the insolation quantities 
                  of Earth. {\it in preparation}



================================================================================


================================================================================
Compilation of the Program
--------------------------

  Edit the makefile to choose a fortran-90 compiler and run "make".




Detailed Description of the Program
-----------------------------------

  insola.par : THE USER should edit the file insola.par to set
          all parameters.

  insola : interactive insolation computations


--------------------------------------------------------------------------------
INSOLA.PAR

The user should edit the parameter file insola.par.

 datefin   : ending   time (Myr)
 datedebut : starting time (Myr)
 pas       : sampling step (in years). 
             The default value is 1000. The minimum step is 1 year
             (it is usually not necessary for pas to be smaller than 100 (yr)). 
 nominsolbin   : Binary intermediate file for insolation
 nomclimapos : ASCII file for climatic variables for positive time  
 nomclimaneg : ASCII file for climatic variables for negative time
 so          : Solar constant expressed in W/m2 
               solar constant at 1 AU so0 =1368 W/m2

 

Starting and ending date must verify :   -19 <= datedebut <= 0 <= datefin <= +9 
The user may restrict himself to e.g. -2Myr to 0 by putting instead
     datedebut=-2.D0
     datefin=0.D0



--------------------------------------------------------------------------------
INSOLA

  The routines insola are designed to compute all
  necessary insolation quantities derived from the orbital and
  precessional quantities computed above.

  They are given on the form of FORTRAN code, so the user can check
  if they correspond to his needs. He can also design his own insolation
  routines.

  Insola will construct a binary intermediate file 
  when this file is not present before computations.

  insola is a self documented program
  For more details, the user can refer to  (Laskar et al, 1993,2004)

================================================================================
User feed-back is encouraged. Unless otherwise specified, send comments and bug 
reports to:                    E-mail     : laskar@imcce.fr
                               Fax        : (33) 1 40 51 20 55
                               Postal mail: Institut de Mecanique Celeste
                                            77 avenue Denfert Rochereau
                                            F-75014 PARIS
================================================================================

