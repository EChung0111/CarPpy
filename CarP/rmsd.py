"""
RMSD calculation, via quaterion-based characteristic polynomial in
pure python/numpy.
Reference
---------
Rapid calculation of RMSDs using a quaternion-based characteristic polynomial.
Acta Crystallogr A 61(4):478-480.
"""
import numpy as np
import math

def _center(conformation):
    """Center and typecheck the conformation"""

    conformation = np.asarray(conformation)
    if not conformation.ndim == 2:
        raise ValueError('conformation must be two dimensional')
    _, three = conformation.shape
    if not three == 3:
        raise ValueError('conformation second dimension must be 3')

    centroid = np.mean(conformation, axis=0)
    centered = conformation - centroid
    return centered

def rmsd_qcp(conformation1, conformation2, rotation_matrix = False):
    """Returns the optimal rotation matrix of two conformers and compute the RMSD with Theobald's quaterion-based characteristic polynomial
    Rapid calculation of RMSDs using a quaternion-based characteristic polynomial. Acta Crystallogr A 61(4):478-480.
    This function is taken from Liu, Pu, Dimitris K. Agrafiotis, and Douglas L. Theobald. "Fast determination of the optimal rotational matrix for macromolecular superpositions." Journal of computational chemistry 31.7 (2010): 1561-1563.
    Things to revise: 
    -The function is copy and paste the determination of max_eigenvalue from the rmsd function
    -returning the identity matrix if all columns of the adj matrix is too small should be dependent on the size of the conformers in question. Rn it is just hardcoded from the original script. This does not impact this function very much because the case is very unlikely and it returns an identity matrix as an indication of failure.

    :param conformation1: (np.ndarray, shape=(n_atoms, 3)) The cartesian coordinates of the first conformation
    :param conformation2: (np.ndarray, shape=(n_atoms, 3)) The cartesian coordinates of the second conformation
    :param rotation_matrix: (bool) parameter that will only return the rotation matrix when set to true. False is the default for the function so it will return the rmsd
    :return: (list) This is a list of floats that indicate the optimal rotation matrix of two conformers. The optimal rotation matrix is the rotation that minimizes rmsd of the two conformers
    """
    # center and typecheck the conformations
    A = _center(conformation1)
    B = _center(conformation2)
    if not A.shape[0] == B.shape[0]:
        raise ValueError('conformation1 and conformation2 must have same number of atoms')
    n_atoms = len(A)

    #the inner product of the structures A and B
    G_A = np.einsum('ij,ij', A, A)
    G_B = np.einsum('ij,ij', B, B)
    #print 'GA', G_A, np.trace(np.dot(A.T, A))
    #print 'GB', G_B, np.trace(np.dot(B.T, B))

    # M is the inner product of the matrices A and B
    M = np.dot(B.T, A)

    # unpack the elements
    Sxx, Sxy, Sxz = M[0, :]
    Syx, Syy, Syz = M[1, :]
    Szx, Szy, Szz = M[2, :]

    # do some intermediate computations to assemble the characteristic
    # polynomial
    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Szz2 = Szz * Szz

    Sxy2 = Sxy * Sxy
    Syz2 = Syz * Syz
    Sxz2 = Sxz * Sxz

    Syx2 = Syx * Syx
    Szy2 = Szy * Szy
    Szx2 = Szx * Szx

    SyzSzymSyySzz2 = 2.0*(Syz*Szy - Syy*Szz)
    Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2

    # two of the coefficients
    C2 = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2)
    C1 = 8.0 * (Sxx*Syz*Szy + Syy*Szx*Sxz + Szz*Sxy*Syx - Sxx*Syy*Szz - Syz*Szx*Sxy - Szy*Syx*Sxz)

    SxzpSzx = Sxz + Szx
    SyzpSzy = Syz + Szy
    SxypSyx = Sxy + Syx
    SyzmSzy = Syz - Szy
    SxzmSzx = Sxz - Szx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy
    Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2

    # the other coefficient
    C0 = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2 \
        + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2) \
        + (-(SxzpSzx)*(SyzmSzy)+(SxymSyx)*(SxxmSyy-Szz)) * (-(SxzmSzx)*(SyzpSzy)+(SxymSyx)*(SxxmSyy+Szz)) \
        + (-(SxzpSzx)*(SyzpSzy)-(SxypSyx)*(SxxpSyy-Szz)) * (-(SxzmSzx)*(SyzmSzy)-(SxypSyx)*(SxxpSyy+Szz)) \
        + (+(SxypSyx)*(SyzpSzy)+(SxzpSzx)*(SxxmSyy+Szz)) * (-(SxymSyx)*(SyzmSzy)+(SxzpSzx)*(SxxpSyy+Szz)) \
        + (+(SxypSyx)*(SyzmSzy)+(SxzmSzx)*(SxxmSyy-Szz)) * (-(SxymSyx)*(SyzpSzy)+(SxzmSzx)*(SxxpSyy-Szz))

    # Newton-Raphson
    E0 = (G_A + G_B) / 2.0
    max_eigenvalue = E0
    for i in range(50):
        old_g = max_eigenvalue
        x2 = max_eigenvalue * max_eigenvalue
        b = (x2 + C2) * max_eigenvalue
        a = b + C1
        delta = ((a * max_eigenvalue + C0)/(2.0 * x2 * max_eigenvalue + b + a))
        max_eigenvalue -= delta
        if abs(max_eigenvalue - old_g) < abs(1e-11 * max_eigenvalue):
            break
    if i >= 50:
        raise ValueError('More than 50 iterations needed.')

    if rotation_matrix == False:
        rmsd = np.sqrt(np.abs(2.0 * (E0 - max_eigenvalue) / n_atoms))
        return rmsd
    else:
        #adjoint matrix calculations
        a11 = SxxpSyy + Szz - max_eigenvalue; 
        a12 = SyzmSzy; 
        a13 = - SxzmSzx; 
        a14 = SxymSyx;

        a21 = SyzmSzy; 
        a22 = SxxmSyy - Szz - max_eigenvalue; 
        a23 = SxypSyx; 
        a24= SxzpSzx;

        a31 = a13; 
        a32 = a23; 
        a33 = Syy - Sxx - Szz - max_eigenvalue; 
        a34 = SyzpSzy;

        a41 = a14; 
        a42 = a24; 
        a43 = a34; 
        a44 = Szz - SxxpSyy - max_eigenvalue;

        a3344_4334 = a33 * a44 - a43 * a34; a3244_4234 = a32 * a44-a42*a34;
        a3243_4233 = a32 * a43 - a42 * a33; a3143_4133 = a31 * a43-a41*a33;
        a3144_4134 = a31 * a44 - a41 * a34; a3142_4132 = a31 * a42-a41*a32;
        q1 =  a22*a3344_4334-a23*a3244_4234+a24*a3243_4233;
        q2 = -a21*a3344_4334+a23*a3144_4134-a24*a3143_4133;
        q3 =  a21*a3244_4234-a22*a3144_4134+a24*a3142_4132;
        q4 = -a21*a3243_4233+a22*a3143_4133-a23*a3142_4132;

        qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

        #The following code tries to calculate another column in the adjoint matrix when the norm of thecurrent column is too small.
        #Usually this block will never be activated.  To be absolutely safe this should beuncommented, but it is most likely unnecessary.

        evecprec = 1e-6;
        evalprec = 1e-11;

        if qsqr < evecprec:
            q1 =  a12*a3344_4334 - a13*a3244_4234 + a14*a3243_4233;
            q2 = -a11*a3344_4334 + a13*a3144_4134 - a14*a3143_4133;
            q3 =  a11*a3244_4234 - a12*a3144_4134 + a14*a3142_4132;
            q4 = -a11*a3243_4233 + a12*a3143_4133 - a13*a3142_4132;
            qsqr = q1*q1 + q2 *q2 + q3*q3+q4*q4;
            if qsqr < evecprec:
                a1324_1423 = a13 * a24 - a14 * a23; a1224_1422 = a12 * a24 - a14 * a22;
                a1223_1322 = a12 * a23 - a13 * a22; a1124_1421 = a11 * a24 - a14 * a21;
                a1123_1321 = a11 * a23 - a13 * a21; a1122_1221 = a11 * a22 - a12 * a21;

                q1 =  a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
                q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
                q3 =  a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
                q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;
                qsqr = q1*q1 + q2 *q2 + q3*q3+q4*q4;

                if qsqr < evecprec:
                    q1 =  a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
                    q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
                    q3 =  a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
                    q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;
                    qsqr = q1*q1 + q2 *q2 + q3*q3 + q4*q4;

                    if qsqr<evecprec:
                        #if qsqr is still too small, return the identity matrix.
                        #!!! I need to fix this, the length of this array should be based on size of the molecule i think??
                        rot=[None]*9
                        rot[0] = rot[4] = rot[8] = 1.0;
                        rot[1] = rot[2] = rot[3] = rot[5] = rot[6] = rot[7] = 0.0;
                        return rot

        normq = math.sqrt(qsqr);
        q1 /= normq;
        q2 /= normq;
        q3 /= normq;
        q4 /= normq;

        a2 = q1 * q1;
        x2 = q2 * q2;
        y2 = q3 * q3;
        z2 = q4 * q4;

        xy = q2 * q3;
        az = q1 * q4;
        zx = q4 * q2;
        ay = q1 * q3;
        yz = q3 * q4;
        ax = q1 * q2;

        row1=[a2 + x2 - y2 - z2,2 * (xy - az),2 * (zx + ay)];
        row2=[2 * (xy + az),a2 - x2 + y2 - z2,2 * (yz - ax)];
        row3=[2 * (zx - ay),2 * (yz + ax),a2 - x2 - y2 + z2];
        rot=[row1,row2,row3];

        np_rotmat = np.array(rot)
        return rmsd, np_rotmat


def test():
    """ This function was just to test the rmsd function. Ignore this. This should just be removed
    """
    A = np.arange(120).reshape(40,3) / 50.0
    B = np.arange(120).reshape(40,3) / 50.0
    A[2,0] += .1

    print (rmsd_qcp(A, B))

if __name__ == '__main__':
    test()
