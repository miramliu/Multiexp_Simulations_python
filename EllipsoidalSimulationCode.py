import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from scipy.stats import rice
from statistics import mean
import random


# So create ellipsoid, and rotate it to create eigenvalues and eigenvectors (i.e. create anisotropic diffusion ellipsoid)
# then find Diffusion tensor (knowing eigenvalues and eigenvectors from the ellipsoid)
# then compare Dxx + Dyy + Dzz to the sum of the eigenvalues? 

#also work in the ROTATED ELLIPSOID PLANE. so as if the rotated 3D orthogonal axes are "x,y,z" 
#the "lab" coordinates (i.e. MRI directions) are then rotated by thetaxyz = [thetax,thetay,thetaz] where they are all orthogonal

'''Rotations: Give angle wanted rotated to respective function, returns rotated point(s).'''

def Rx(x):
    Rx = np.matrix([[1,0,0],[0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return Rx
def Ry(y):
    Ry = np.matrix([[np.cos(y),0,np.sin(y)],[0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return Ry
def Rz(z):
    Rz = np.matrix([[np.cos(z), - np.sin(z), 0],[np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz
def Rxyz (thet):
    Rxyz = Rx(thet[0])*Ry(thet[1])*Rz(thet[2])
    return np.array(Rxyz)


#defines ellipsoid given coefficients a,b,c, center [0,0,0], and give rotation thetaxyz
def ellipsoid(abc,thetaxyz):
    a = abc[0]
    b = abc[1]
    c = abc[2]
    A = np.array([[a,0,0],[0,b,0],[0,0,c]])
    center = [0,0,0]
    eigenvectors,eigenvalues,rotated = np.linalg.svd(A) #ignore rotation as it's made in standard xyz, then will be rotated
    theta = np.linspace(0.0, np.pi, 100)
    phi = np.linspace(0.0, 2.0 * np.pi, 100)
    x = eigenvalues[0] * np.outer(np.cos(theta), np.sin(phi))
    y = eigenvalues[1] * np.outer(np.sin(theta), np.sin(phi))
    z = eigenvalues[2] * np.outer(np.ones_like(theta), np.cos(phi))

    rotation = Rxyz(thetaxyz)
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    return x,y,z

    
#create 3 orthogonal vectors, rotated by thetaxyz = [theta_x, theta_y, theta_z]
def OrthogVects(thetaxyz):
    xhat = [1,0,0]
    yhat = [0,1,0]
    zhat = [0,0,1]
    
    xrotated = Rxyz(thetaxyz)*xhat
    yrotated = Rxyz(thetaxyz)*yhat
    zrotated = Rxyz(thetaxyz)*zhat
    return xrotated, yrotated, zrotated
#this is jst trying to take the standard orthonormal vectors and rotate it by x,y,z around x and y and z axis... 


#this is from stackoverflow to get equal axes...https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return 


'''make_line (ML): given a point, vector, and length, makes the corresponding line '''
def ML(p,v,L):
    pointL = p
    VectL = numpy.array(v)
    Lwant = int(L)
    VectLNorm = N(v)
    t = numpy.linspace(0,Lwant,50) #make related to wanted length??
    x = [pointL[0]]
    y = [pointL[1]]
    z = [pointL[2]]
    for t in range (0,Lwant):
        L = numpy.sqrt(((VectLNorm[0]*t)**2 + (VectLNorm[2]*t)**2 + (VectLNorm[2]*t)**2))
        xL = pointL[0] + t*VectLNorm[0]
        yL = pointL[1] + t*VectLNorm[1]
        zL = pointL[2] + t*VectLNorm[2]
        if L <= Lwant:
            x.append(xL)
            y.append(yL)
            z.append(zL)
    return [x,y,z]
    


#get the average diffusion from the primary diffusion ellipsoid axes (this is lambda 1 + lambda 2 + lambda 3 divided by 3
#this can only be gotten from DTI...multiple directions.
#then rotate ellipsoid into 'lab frame' annd get the new sum of axes
#this is then Dxx + Dyy + Dzz divided by 3. 
def CompareAxes(abc,thetaxyz):
    a = abc[0]
    b = abc[1]
    c = abc[2]
    #set up innitial 3d ellipsoid
    A = np.array([[a,0,0],[0,b,0],[0,0,c]])
    center = [0,0,0]
    eigenvectors,eigenvalues,rotated = np.linalg.svd(A) #ignore 'rotated' as it's made in standard xyz, then will be rotated
    theta = np.linspace(0.0, np.pi, 100) #cover all angles to build ellipsoid
    phi = np.linspace(0.0, 2.0 * np.pi, 100)  #cover all angles to build ellipsoid
    #get major axes
    x = eigenvalues[0] * np.outer(np.cos(theta), np.sin(phi))
    y = eigenvalues[1] * np.outer(np.sin(theta), np.sin(phi))
    z = eigenvalues[2] * np.outer(np.ones_like(theta), np.cos(phi))
    
    #rotate by thetaxyz
    rotation = Rxyz(thetaxyz)
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    #now maybe just find the values here that are closest to the xyz LAB FRAME axes? 
    ###
    # gotta fix the meshgrid stuff... 
    Dxx,Dyy,Dzz = find_axes(x,y,z) #just get the length of the ellipsoid along the normal axis (i.e. 3 random orthogonal directions)
    ADC = (Dxx + Dyy + Dzz)/3
    DTI = (a + b + c)/3 #compare to the actual eigenvectors

    return ADC,DTI

#find the index where two values of array are closest to zero (to find axes)
#assumes x and y and z are meshgrid (2D)
def find_axes(array1, array2, array3):
    value = 0 #closest to 0
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    idx1,idx2 = findminidx(np.abs(array2)+np.abs(array3)) #first axis (along array1)
    Xval = np.abs(array1[idx1,idx2])
    idx1,idx2 = findminidx(np.abs(array1)+np.abs(array3)) #second axis (along array2)
    Yval = np.abs(array2[idx1,idx2])
    idx1,idx2 = findminidx(np.abs(array1)+np.abs(array2)) #third axis (along array3)
    Zval = np.abs(array3[idx1,idx2])
    return Xval, Yval, Zval

def findminidx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k//ncol, k%ncol


def Calc_FA(a,b,c):
    MD = (a+b+c)/3
    num = np.sqrt((a-MD)**2 + (b-MD)**2 + (c-MD)**2)
    denom = np.sqrt(a**2 + b**2 + c**2)
    return np.sqrt(3/2)*num/denom

def Calc_FA(a,b,c):
    MD = (a+b+c)/3
    num = np.sqrt((a-MD)**2 + (b-MD)**2 + (c-MD)**2)
    denom = np.sqrt(a**2 + b**2 + c**2)
    return np.sqrt(3/2)*num/denom


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def GenerateFA_Ellipsoid(FA_want,FAs,abcs):
    val,idx = find_nearest(FAs,FA_want)
    return abcs[idx], val#, FA(abcs[idx][0],abcs[idx][1],abcs[idx][2])
    

# NEW RICIAN FUNCTION.From old code. double check calculation of SNSR and sigma? 
#SNR = 1, sigma = 1
#SNR = 10, sigma = .1
#SNR = 20, sigma = .045
#SNR = 50, sigma = .02
#SNR = 75, sigma = .013
#SNR = 100, sigma = .01
#SNR = 125, sigma = .008
#SNR = 150, sigma = .0065
def NoiseRice(I,sigma): # noise with rician distribution
    N = [] 
    #v = .79 # calculated from images on 6/11/19
    #sigma = .013 #(std of noise measured!)
    #v = 0.00434 #scaled = .79/182
    v = .005816 #scaled 6/27/19
    b = v/sigma
    r = rice.rvs(b, scale = sigma, size=len(I))
    for i in range(0,len(I)):
        N.append(I[i] +r[i]) #SNR = 1,10,22,100,150,inf
    return N

def triexp_func(b, frac_fast,frac_med,frac_slow,diff_fast,diff_med,diff_slow):
    Data = frac_slow*np.exp(-b*diff_slow)+frac_med*np.exp(-b*diff_med)+frac_fast*np.exp(-b*diff_fast)
    normal = frac_slow+frac_med+frac_fast
    return Data/normal

def Scale_Ds(D_traces,abc_s,thetas):
    D_thetas = np.zeros(len(D_traces))
    for n in range(len(D_traces)): #here n is fast, medium, or slow compartments. i.e. the 3 ellipsoids
        D_n = 3*D_traces[n]/sum(abc_s[n]) #normalize to ellipsoid (basically normalize to 1)
        D_thetas[n] = D_n*thetas[n] #multiply by diffusion strength along b-value direction 
    return D_thetas #return the three single-values multiplied by Dn scale



#create an ellipsoid with a given FA
#rotate it within the lab frame (i.e. slow compartment is tilted k-degrees in lab frame while fast compartment is tilted h-degrees in lab frame)
def generate_rotated_anisotropic_ellipsoid(FA,compartment_rotations):
    # FAs and abcs are just previously generated potentials to choose from
    FAs = np.zeros(100)
    abcs = np.empty([100,3])
    for j in range(1,100): #creating an elipsoid
        abc3 = [1,random.uniform(0.01,1),random.uniform(0.01,1)]
        #print(abc3)
        FAs[j] = (Calc_FA(abc3[0],abc3[1],abc3[2]))
        abcs[j,:] = abc3
        #print(abc3)
    abc, nearest_FA = GenerateFA_Ellipsoid(FA,FAs,abcs) #getting the ellipsoidal dimensions
    if sum(abc)< 0.00000001: # ifi it's very small... set it to 1,1,1 (isotropic.) Nov 14 2024
        abc=[1,1,1]
    a = abc[0] # i.e. Ax
    b = abc[1] # i.e. By
    c = abc[2] # i.e. Cz
    #set up innitial 3d ellipsoid
    A = np.array([[a,0,0],[0,b,0],[0,0,c]])
    center = [0,0,0]
    eigenvectors,eigenvalues,rotated = np.linalg.svd(A) #ignore 'rotated' as it's made in standard xyz, then will be rotated
    theta = np.linspace(0.0, np.pi, 100) #cover all angles to build ellipsoid
    phi = np.linspace(0.0, 2.0 * np.pi, 100)  #cover all angles to build ellipsoid
    #get major axes
    x = eigenvalues[0] * np.outer(np.cos(theta), np.sin(phi))
    y = eigenvalues[1] * np.outer(np.sin(theta), np.sin(phi))
    z = eigenvalues[2] * np.outer(np.ones_like(theta), np.cos(phi))
    
    #rotate by compartment_dependent lab-frame angles
    rotation = Rxyz(compartment_rotations)
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    return x,y,z,abc,nearest_FA


#give all fractional anisotropies and their lab-frame rotations to generate the three compartment ellipsoids up front
def CreateThreeEllipsoidalCompartments(All_FAs, All_rotations):
    FA_fast, FA_med, FA_slow = All_FAs[0],All_FAs[1], All_FAs[2] # the anisotropies
    fastrot,medrot,slowrot = All_rotations[0], All_rotations[1], All_rotations[2] #the original ellipsoid rotations (in the lab frame, so not all aligned in same direction)
    
    #create the three compartment ellipsoids rotated in lab-space, and then rotate along the b-value direction
    x_fast,y_fast,z_fast,abc_fast,nearest_FA_fast = generate_rotated_anisotropic_ellipsoid(FA_fast,fastrot) # but then thetaxyz are same for fast, med, slow, because it's measured in the same lab-frame direction
    x_med,y_med,z_med,abc_med, nearest_FA_med = generate_rotated_anisotropic_ellipsoid(FA_med,medrot)
    x_slow,y_slow,z_slow,abc_slow, nearest_FA_slow = generate_rotated_anisotropic_ellipsoid(FA_slow,slowrot)
    
    fast_compartment = [x_fast,y_fast,z_fast,abc_fast]
    med_compartment = [x_med,y_med,z_med,abc_med]
    slow_compartment = [x_slow,y_slow,z_slow,abc_slow]
    nearest_FAs = [nearest_FA_fast,nearest_FA_med,nearest_FA_slow]
    return fast_compartment, med_compartment, slow_compartment, nearest_FAs

def RotateCompartments(x,y,z,thetaxyz):
    center = [0,0,0]
    rotation = Rxyz(thetaxyz)
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    return x,y,z

#give true D, the three ellipsoidal compartments in labframe, and the b-value direction (i.e. thetaxyz)
def GetOrthogonalD_thetas(D_traces,fast_compartment, med_compartment, slow_compartment):
    
    #if rotating to measure in different b-value directions. 
    #now "measured along 3 orthogonal diffusion directions" to get 3 single-direction b-values
    #rotate by thetaxyz ### NO, NO THETAXYZ BECAUSE EVERY B-VAL IS ALONG THE SAME 3 DIRECTIONS, so rotate each individual ellipsoid, but measure along same 3 directions.
    #x_fast,y_fast,z_fast = RotateCompartments(fast_compartment[0],fast_compartment[1],fast_compartment[2],thetaxyz)
    #x_med,y_med,z_med = RotateCompartments(med_compartment[0],med_compartment[1],med_compartment[2],thetaxyz)
    #x_slow,y_slow,z_slow = RotateCompartments(slow_compartment[0],slow_compartment[1],slow_compartment[2],thetaxyz)

    #now measure along the given b-value direction
    x_theta_fast,y_theta_fast,z_theta_fast = find_axes(fast_compartment[0],fast_compartment[1],fast_compartment[2]) #just get the length of the ellipsoid along the normal axis (i.e. 3 random orthogonal directions)
    x_theta_med,y_theta_med,z_theta_med = find_axes(med_compartment[0],med_compartment[1],med_compartment[2]) #just get the length of the ellipsoid along the normal axis (i.e. 3 random orthogonal directions)
    x_theta_slow,y_theta_slow,z_theta_slow = find_axes(slow_compartment[0],slow_compartment[1],slow_compartment[2]) #just get the length of the ellipsoid along the normal axis (i.e. 3 random orthogonal directions)

    #now assume the single direction is measured along the x-axis (just so it's one direction)
    x_thetas = [x_theta_fast, x_theta_med, x_theta_slow]
    abc_s = [fast_compartment[3],med_compartment[3],slow_compartment[3]]
    #get the new "diffusion values" i.e. the AxD_slow, AxD_med, AxD_fast
    D_traces_b = Scale_Ds(D_traces,abc_s,x_thetas) 
    
    #now for comparison, also exporting the averaged across 3 values
    fast_averaged = mean([x_theta_fast,y_theta_fast,z_theta_fast])
    med_averaged = mean([x_theta_med,y_theta_med,z_theta_med])
    slow_averaged = mean([x_theta_slow,y_theta_slow,z_theta_slow])
    averaged_thetas = [fast_averaged, med_averaged, slow_averaged]
    D_traces_b_averaged = Scale_Ds(D_traces,abc_s,averaged_thetas)

    return D_traces_b,D_traces_b_averaged


    
    