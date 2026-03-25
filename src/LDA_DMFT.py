import shutil
import pyalps.mpi as mpi                # mpi library
from pyalps.hdf5 import h5ar            # hdf5 interface
import pyalps.cthyb as cthyb            # the solver module
from numpy import sqrt,cosh,sinh,exp,pi #some math
from numpy import array,zeros,append
import numpy as np
from math import fabs
import scipy.optimize
from sys import exit
#import ctypes
#ctypes.CDLL('libmkl_rt.so', ctypes.RTLD_GLOBAL)
##################################################################################################################
#                                                                                                                #
#                                               P A R A M E T E R S                                              #
#                                                                                                                #
##################################################################################################################

# This script takes some time to run to get converged results. Make sure the runtime is large enough to get
# sensible results (depends on the number of processes).
runtime_dmft=2000       # 1500 runtime for each DMFT iteration
runtime_dmft_final=5000 # 6000 increase runtime on final iteration for additional measurements
dmft_iterations=13     # number of DMFT iterations

# perform calculations for fixed U. 

mu=0.000
previous_present=1
# for simplicity, this script is for a single set of parameters only
# we list all solver parameters here for completeness
parms = {
# solver parameters
# general
'SWEEPS'                     : 1000000000,                         #sweeps to be done
'THERMALIZATION'             : 2000,                               #thermalization sweeps to be done
'SEED'                       : 42,                                 #random number seed
'N_MEAS'                     : 1000,                                #number of sweeps after which a measurement is done
'N_ORBITALS'                 : 6,                                  #number of 'orbitals', i.e. number of spin-orbital degrees of freedom or segments
'BASENAME'                   : "hyb.param",                        #base name of the h5 output file
'MAX_TIME'                   : runtime_dmft,                       #runtime of the solver per iteration
'VERBOSE'                    : 1,                                  #whether to output extra information
'COMPUTE_VERTEX'             : 0,                                  #whether to compute the vertex function
'TEXT_OUTPUT'                : 1,                                  #whether to write results in human readable (text) format
# file names
'DELTA'                      : "Delta.h5",                         #file name of the hybridization function
'DELTA_IN_HDF5'              : 1,
#'U_MATRIX'                   : "U_matrix.dat",                                   #whether to read the hybridization from an h5 archive
'U'                          : 2.30,
'J'                          : 0.17,
'MU_VECTOR'                  : "muvector.dat",                #chemical potential (MU=U/2-2K'(0) corresponds to half-filling; here K(0)=Lambda^2/w0)
'BETA'                       : 60.0,                               #inverse temperature
# measurements
'MEASURE_freq'               : 1,                                  #whether to measure single-particle Green's function on Matsubara frequencies
'MEASURE_legendre'           : 1,                                  #whether to measure single-particle Green's function in Legendre polynomial basis
'MEASURE_g2w'                : 0,                                  #whether to measure two-particle Green's function on Matsubara frequencies
'MEASURE_h2w'                : 0,                                  #whether to measure the higher-order correlation function for the vertex on Matsubara frequencies
'MEASURE_nn'                 : 0,                                  #whether to measure equal-time density-density correlations
'MEASURE_nnt'                : 0,                                  #whether to measure the density-density correlation function (local susceptibility) in imaginary time
'MEASURE_nnw'                : 0,                                  #whether to measure the density-density correlation function (local susceptibility) on Matsubara frequencies
'MEASURE_sector_statistics'  : 0,                                  #whether to measure sector statistics
# measurement parameters
'N_HISTOGRAM_ORDERS'         : 100,                                 #maximum order for the perturbation order histogram
'N_TAU'                      : 2000,                               #number of imaginary time points (tau_0=0, tau_N_TAU=BETA)
'N_MATSUBARA'                : 512,   
'N_nn'                       : 1000,                               #number of imaginary time points for the density-density correlation function
'N_W'                        : 500,
'N_LEGENDRE'                 : 125,                                 #number of Legendre coefficients
# additional parameters (used outside the solver only)
't'                          : 0.0,
'SPINFLIP'                   : 1,              #hopping
'mix'                         : 0.5                                 #mixing parameter for hybridization update
}# parms

if mpi.rank==0:
######################################################################
######################################################################


   I=complex(0.,1.)
 

   
#######################################################################
#######################################################################
#  H_k Generation

   def Generate_Hk():
      m=[]

      for i in open('Hk_pyth'):
          m.append(i)
      s=len(m)

      k=[]
      y=[]
      f=open('Hk_pyth','r')
      for i in range(s):
          x=f.readline()
          if(i==0):
           y=x.split()
           No_kpoints=int(y[0])
           No_orbitals=int(y[1])
          else:
           y=x.split()
           g=[eval(i) for i in y]
           k.append(complex(g[0],g[1]))
           
      a=np.array(k,dtype=complex,ndmin=1)

      a=a.reshape((No_kpoints,No_orbitals,No_orbitals))

      b=a

      c=np.zeros((No_orbitals,No_orbitals),dtype=complex)

      d=c

      Hk=np.zeros((No_kpoints,2*No_orbitals,2*No_orbitals),dtype=complex)

      for i in range(No_kpoints):
             f=a[i]
             g=b[i]
             Hk[i]=np.bmat([[f,c],[d,g]])
 
      return Hk


########################################################################
########################################################################
# Initialization

   size=8000
   Hk_f=np.zeros((size,6,6),dtype=complex)
   iw=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
   iw_imp=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
   sigma=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
   Gw=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
   hyb_w=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
   beta=parms['BETA']
   N_mastu=parms["N_MATSUBARA"]
   N_tau=parms["N_TAU"]
   Delta_tau=np.zeros((parms["N_TAU"]+1,6,6),dtype=float)
   
########################################################################
########################################################################
# Generate matsubar frequency grid


   for i in range(parms["N_MATSUBARA"]):
     for j in range(6):
      iw[i,j,j]=I*(((2.0*i)+1)*np.pi)/beta
#     for k in range(6):
#      iw_imp[i,k,k]=I*(((2.0*i)+1)*np.pi)/beta


########################################################################
########################################################################
# DC correction
   
   DC_full=np.zeros((6,6),dtype=complex) 
   DC_imp=np.zeros((6,6),dtype=complex)
#   for j in range(6):
#        DC_full[j,j]=0.0
#   for n in range(6):
#        DC_imp[n,n]=0.0     

 
########################################################################
#######################################################################
# Calculatin of Greens function and hyb in Matsubara frequency 

   def Img_Green(N_mastu1,size1,mu1,mom1,DC1,iw1,y1,sigma1):
     Gw1=np.zeros((N_mastu1,6,6),dtype=complex)
     Iden=np.eye(6,dtype=complex)
     for i in range(N_mastu1):
       sum1=np.zeros((6,6),dtype=complex)
       for j in range(size1):
           sum1=sum1+np.linalg.inv(iw1[i]+(Iden*mu1)-DC1-y1[j]-sigma1[i])
       Gw1[i]=sum1/size1
       hyb_w[i]=iw1[i]+(Iden*mu1)-mom1-DC1-np.linalg.inv(Gw1[i])-sigma1[i]
     return hyb_w

#   def Construct_imp_G(N_mastu1,size1,mu1,mom1,DC1,iw1,sigma1):
#      Iden=np.eye(6,dtype=complex)
#      Imp_G=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
#      Imp_self=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
#      for i in range(N_mastu1):  
#         Imp_G[i]=Gw[i]
#         Imp_self[i]=sigma1[i]
#         hyb_w[i]=iw1[i]+(Iden*mu1)-mom1-DC1-np.linalg.inv(Imp_G[i])-Imp_self[i]

#      return hyb_w    

############################################################################
############################################################################
# Calculaton of Delta in tau space.

   def Tau_green(N_mastu1,N_tau1,beta1,mom1,mom2,iw1,hyb1):
     
      Tau_grid=np.zeros((N_tau1+1),dtype=float)
      Iden=np.eye(6,dtype=float)
      I=complex(0.,1.)
      for i in range(N_tau1+1):

        Tau_grid[i]=(i*beta1)/N_tau1
        sum1=np.zeros((6,6),dtype=float)
        sum4=np.zeros((N_mastu1),dtype=complex)
        sum3=np.zeros((N_mastu1,6,6),dtype=complex)
        for j in range(N_mastu1):


          sum4[j]=np.exp(-I*np.imag(iw1[j,0,0])*Tau_grid[i])
          sum3[j]=(hyb1[j]-((mom2-mom1)*np.linalg.inv(iw1[j])))*sum4[j]
          sum1=sum1+np.real(sum3[j])
        Delta_tau[i]=sum1*(2.0/beta1)
        Delta_tau[i]=Delta_tau[i]-(np.real(0.5*(mom2-mom1)))
                  
      return Delta_tau

####################################################################################
####################################################################################
# Calculation of net_density

   def density(N_mastu1,size1,mu1,DC1,beta1,iw1,y1,sigma1):  
        
     Iden=np.eye(6,dtype=float)
     density1=np.zeros((N_mastu1,6,6),dtype=float)
     Green1=np.zeros((N_mastu1,6,6),dtype=complex)
     density2=np.zeros((6,6),dtype=float)
     for i in range(N_mastu1):
       sum1=np.zeros((6,6),dtype=complex)
       for j in range(size1):
           sum1=sum1+np.linalg.inv(iw1[i]+(Iden*mu1)-y1[j]-DC1-sigma1[i])
       Green1[i]=sum1/size1
       density1[i]=np.real(Green1[i]-np.linalg.inv(iw1[i]))
     density2=density1.sum(axis=0)
     density2=density2*(2.0/beta1)
     density2=density2+(0.50*Iden)
     final_den=np.trace(density2)
     print final_den,mu1
     return final_den #,imp_occu
#####################################################################################
#####################################################################################
# Find mu by fixing density
     
   def F(x):
      y=0.0;
      n_tot=3.858;
      mu=x[0]
      y=density(N_mastu,size,mu,DC_full,beta,iw,Hk_f,sigma)
      
      return  n_tot-y


#######################################################################################
#######################################################################################
     
 



   Hk_f=Generate_Hk()
#########################################################################
#########################################################################   
# calculate moments
   def moments_cal(Hk_f,size1):
      second_mom1=np.zeros((6,6),dtype=complex)     
      for j in range(size1):

       second_mom1=second_mom1+np.dot(Hk_f[j],Hk_f[j])/size1
      return second_mom1
   second_mom=np.zeros((6,6),dtype=complex)
   first_mom=Hk_f.sum(axis=0)/size
   prod_mom=np.dot(first_mom,first_mom)
   second_mom=moments_cal(Hk_f,size)
############################################################################
############################################################################
   if(previous_present==0):
    hyb_w=Img_Green(N_mastu,size,mu,first_mom,DC_full,iw,Hk_f,sigma)
    
#    hyb_w=Construct_imp_G(N_mastu,size,mu,first_mom,DC_imp,iw_imp,sigma)
    
    Delta_tau=Tau_green(N_mastu,N_tau,beta,prod_mom,second_mom,iw,hyb_w)



   
    delta0=np.zeros(N_tau+1,dtype=float)
    delta1=np.zeros(N_tau+1,dtype=float)
    delta2=np.zeros(N_tau+1,dtype=float)
    delta3=np.zeros(N_tau+1,dtype=float)
    delta4=np.zeros(N_tau+1,dtype=float)
    delta5=np.zeros(N_tau+1,dtype=float)
    delta6=np.zeros(N_tau+1,dtype=float)

    for i in range(N_tau+1):

        delta0[i]=-np.absolute(Delta_tau[i,0,0])
        delta1[i]=-np.absolute(Delta_tau[i,1,1])
        delta2[i]=-np.absolute(Delta_tau[i,2,2])
        delta3[i]=-np.absolute(Delta_tau[i,3,3])
        delta4[i]=-np.absolute(Delta_tau[i,4,4])
        delta5[i]=-np.absolute(Delta_tau[i,5,5])


    
    ar=h5ar(parms['DELTA'],'w')
    for m in range(parms["N_ORBITALS"]):
       if(m==0):
        ar['/Delta_%i'%m]=delta0
       elif(m==1):
        ar['/Delta_%i'%m]=delta3
       elif(m==2):
        ar['/Delta_%i'%m]=delta1
       elif(m==3):
        ar['/Delta_%i'%m]=delta4
       elif(m==4):
        ar['/Delta_%i'%m]=delta2
       else:
        ar['/Delta_%i'%m]=delta5

    del ar    


    f=open("muvector.dat","w")
    for m in range(parms["N_ORBITALS"]):
        if(m==0):
          f.write("%f "%(mu-np.real(first_mom[0,0])-np.real(DC_imp[0,0])))
        elif(m==1):
          f.write("%f "%(mu-np.real(first_mom[3,3])-np.real(DC_imp[3,3])))
        elif(m==2):
          f.write("%f "%(mu-np.real(first_mom[1,1])-np.real(DC_imp[1,1])))
        elif(m==3):
          f.write("%f "%(mu-np.real(first_mom[4,4])-np.real(DC_imp[4,4])))
        elif(m==4):
          f.write("%f "%(mu-np.real(first_mom[2,2])-np.real(DC_imp[2,2])))
        else:
          f.write("%f "%(mu-np.real(first_mom[5,5])-np.real(DC_imp[5,5])))


    f.close()
 

   else:
     
     f=open("mu_guess.dat","r")
     mu1=f.read()
     mu=float(mu1)
     f.close()

     f=open("muvector.dat","w")
     for m in range(parms["N_ORBITALS"]):
        if(m==0):
          f.write("%f "%(mu-np.real(first_mom[0,0])-np.real(DC_imp[0,0])))
        elif(m==1):
          f.write("%f "%(mu-np.real(first_mom[3,3])-np.real(DC_imp[3,3])))
        elif(m==2):
          f.write("%f "%(mu-np.real(first_mom[1,1])-np.real(DC_imp[1,1])))
        elif(m==3):
          f.write("%f "%(mu-np.real(first_mom[4,4])-np.real(DC_imp[4,4])))
        elif(m==4):
          f.write("%f "%(mu-np.real(first_mom[2,2])-np.real(DC_imp[2,2])))
        else:
          f.write("%f "%(mu-np.real(first_mom[5,5])-np.real(DC_imp[5,5])))


     f.close()
  


     ar=h5ar(parms['DELTA'],'w')
     for m in range(parms['N_ORBITALS']):
       delta_old=ar['/Delta_%i'%m]
       ar['/Delta_%i'%m]=delta_old
     del ar

mpi.world.barrier() # wait until solver input is written to file

###################################################################################################################
#                                                                                                                 #
#                                D M F T   S E L F C O N S I S T E N C Y    L O O P                               #
#                                                                                                                 #
###################################################################################################################
for it in range(dmft_iterations):

  if mpi.rank==0:
    print "****************************************************************************"
    print "*                           DMFT iteration %3i                             *"%(it)
    print "****************************************************************************"

  # !always make sure that parameters are changed on all threads equally!
  # (i.e. don't wrap this into an 'if mpi.rank==0' statement)
  if it==dmft_iterations-1:
    parms['MAX_TIME'] = runtime_dmft_final
    # turn on additional measurements for the final dmft interation
    parms['MEASURE_nn']=1
    parms['MEASURE_nnt']=1
    parms['MEASURE_nnw']=1
    parms['MEASURE_sector_statistics']=1
    parms['TEXT_OUTPUT']=1  # this will write results of the final iteration in text format

  # write parameters for reference (on master only)
  if mpi.rank==0:
    ar=h5ar(parms['BASENAME']+'.h5','a')
    ar['/parameters']=parms
    ar['/parameters%i'%it]=parms # this is a backup for each iteration
    del ar

  # solve the impurity model in parallel
  cthyb.solve(parms)

  # self-consistency on the master
  if mpi.rank==0:
    if(parms['TEXT_OUTPUT']==1):
      shutil.copy("Gt.dat", "Gt%i.dat"%it)
      shutil.copy("Sw.dat", "Sw%i.dat"%it)
      shutil.copy("simulation.dat", "simulation%i.dat"%it) # keep some basic information for each iteration
                                                                              # read Green's function from file
    ar=h5ar(parms['BASENAME']+'.out.h5','rw')

    sigma_0=np.zeros(parms['N_MATSUBARA'],dtype=complex)
    sigma_1=np.zeros(parms['N_MATSUBARA'],dtype=complex)
    sigma_2=np.zeros(parms['N_MATSUBARA'],dtype=complex)
    sigma_3=np.zeros(parms['N_MATSUBARA'],dtype=complex)
    sigma_4=np.zeros(parms['N_MATSUBARA'],dtype=complex)
    sigma_5=np.zeros(parms['N_MATSUBARA'],dtype=complex)

    sigma0=ar['/S_l_omega/0/mean/value']
    sigma1=ar['/S_l_omega/1/mean/value']
    sigma2=ar['/S_l_omega/2/mean/value']
    sigma3=ar['/S_l_omega/3/mean/value']
    sigma4=ar['/S_l_omega/4/mean/value']
    sigma5=ar['/S_l_omega/5/mean/value']


    sigma=np.zeros((parms["N_MATSUBARA"],6,6),dtype=complex)
    for i in range(parms['N_MATSUBARA']):
         sigma[i,0,0]=sigma0[i]  
         sigma[i,1,1]=sigma2[i]
         sigma[i,2,2]=sigma4[i]
         sigma[i,3,3]=sigma1[i]
         sigma[i,4,4]=sigma3[i]
         sigma[i,5,5]=sigma5[i]

    x = scipy.optimize.broyden1(F, [mu], f_tol=1e-6)
    mu=x[0]
   
    hyb_w=Img_Green(N_mastu,size,mu,first_mom,DC_full,iw,Hk_f,sigma)

#    hyb_w=Construct_imp_G(N_mastu,size,mu,first_mom,DC_imp,iw_imp,sigma)

    Delta_tau=Tau_green(N_mastu,N_tau,beta,prod_mom,second_mom,iw,hyb_w)
 
   
#    hyb_w=Img_Green(N_mastu,size,mu,DC_full,iw,Hk_f,sigma)
#    hyb_w=Construct_imp_G(N_mastu,size,mu,imp_first_mom,DC_imp,iw_imp,sigma)


#    Delta_tau=Tau_green(N_mastu,N_tau,beta,imp_prod_mom,imp_second_mom,iw_imp,hyb_w)



    delta0=np.zeros(N_tau+1,dtype=float)
    delta1=np.zeros(N_tau+1,dtype=float)
    delta2=np.zeros(N_tau+1,dtype=float)
    delta3=np.zeros(N_tau+1,dtype=float)
    delta4=np.zeros(N_tau+1,dtype=float)
    delta5=np.zeros(N_tau+1,dtype=float)

    for i in range(N_tau+1):

        delta0[i]=-np.absolute(Delta_tau[i,0,0])
        delta1[i]=-np.absolute(Delta_tau[i,1,1])
        delta2[i]=-np.absolute(Delta_tau[i,2,2])
        delta3[i]=-np.absolute(Delta_tau[i,3,3])
        delta4[i]=-np.absolute(Delta_tau[i,4,4])
        delta5[i]=-np.absolute(Delta_tau[i,5,5])



    for m in range(parms["N_ORBITALS"]):
       if(m==0):
        ar['/G_tau_new_%i'%m]=delta0
       elif(m==1):
        ar['/G_tau_new_%i'%m]=delta3
       elif(m==2):
        ar['/G_tau_new_%i'%m]=delta1
       elif(m==3):
        ar['/G_tau_new_%i'%m]=delta4
       elif(m==4):
        ar['/G_tau_new_%i'%m]=delta2
       else:
        ar['/G_tau_new_%i'%m]=delta5
       
    del ar


   
    f=open("muvector.dat","w")
    for m in range(parms["N_ORBITALS"]):
        if(m==0):
          f.write("%f "%(mu-np.real(first_mom[0,0])-np.real(DC_imp[0,0])))
        elif(m==1):
          f.write("%f "%(mu-np.real(first_mom[3,3])-np.real(DC_imp[3,3])))
        elif(m==2):
          f.write("%f "%(mu-np.real(first_mom[1,1])-np.real(DC_imp[1,1])))
        elif(m==3):
          f.write("%f "%(mu-np.real(first_mom[4,4])-np.real(DC_imp[4,4])))
        elif(m==4):
          f.write("%f "%(mu-np.real(first_mom[2,2])-np.real(DC_imp[2,2])))
        else:
          f.write("%f "%(mu-np.real(first_mom[5,5])-np.real(DC_imp[5,5])))
    f.close()
 
    f=open("mu_guess.dat","w")
    f.write("%f "%(mu))
    f.close()
 



    ar=h5ar(parms['BASENAME']+'.out.h5','r')
    gt=np.zeros(N_tau+1,dtype=float)
    for m in range(parms['N_ORBITALS']):
        gt+=ar['G_tau_new_%i'%m]
    gt/=6.0
    del ar

    ar=h5ar(parms['DELTA'],'w')
    for m in range(parms['N_ORBITALS']):
           ar['/Green_%i'%m]=gt
    del ar
    ar=h5ar(parms['DELTA'],'rw')
    for m in range(parms['N_ORBITALS']):
      gt=np.zeros(N_tau+1,dtype=float)
      delta_new=np.zeros(N_tau+1,dtype=float)
      delta_old=ar['/Delta_%i'%m]
      gt=ar['/Green_%i'%m]
      
      delta_new=(1.-parms['mix'])* gt + parms['mix']*delta_old # mix old and new delta
      ar['/Delta_%i'%m]=delta_new
    del ar

  mpi.world.barrier()






    


