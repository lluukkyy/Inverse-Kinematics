#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
    // Students should implement this.
    // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
    // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
    // Then, implement the same algorithm into this function. To do so,
    // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
    // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
    // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
    // so that code is only written once. We considered this; but it is actually not easily doable.
    // If you find a good approach, feel free to document it in the README file, for extra credit

//    for (int i=0; i<numIKJoints; i++){
//        Vec4d handle = fk.getJointGlobalTransform(IKJointIDs[i]) * Vec4d(0,0,0,1);
//        handlePositions[i*3+0] = handle[0];
//        handlePositions[i*3+1] = handle[1];
//        handlePositions[i*3+2] = handle[2];
//    }
//
        int numJoints = fk.getNumJoints();
        Mat3<real> localTransformsR[numJoints];
        Vec3<real> localTransformsT[numJoints];
        Mat3<real> globalTransformsR[numJoints];
        Vec3<real> globalTransformsT[numJoints];
        for(int i=0; i<numJoints; i++)
        {
            real angle[3] = {eulerAngles[i*3+0],eulerAngles[i*3+1],eulerAngles[i*3+2]};
            Mat3<real> angleR = Euler2Rotation(angle,fk.getJointRotateOrder(i));
            real orientation[3] = {fk.getJointOrient(i)[0],fk.getJointOrient(i)[1],fk.getJointOrient(i)[2]};
            Mat3<real> orientationR = Euler2Rotation(orientation,fk.getJointRotateOrder(i));
            localTransformsR[i] = orientationR * angleR;
            localTransformsT[i] = {fk.getJointRestTranslation(i)[0],fk.getJointRestTranslation(i)[1],fk.getJointRestTranslation(i)[2]};
        }

        globalTransformsR[0] = localTransformsR[0];
        globalTransformsT[0] = localTransformsT[0];
        for(int i=1; i<numJoints; i++){
            int index = fk.getJointUpdateOrder(i);
            // given two transforms f1(p) = R1 p + t1 and f2(p) = R2 p + t2,
            // compute the composite transform f1(f2(p)) = R1 R2 p + (R1 t2 + t1)
            // rout = r1 * r2; tout = r1 * t2 + t1;
            globalTransformsR[index] = globalTransformsR[fk.getJointParent(index)]  * localTransformsR[index];
            globalTransformsT[index] = globalTransformsR[fk.getJointParent(index)] * localTransformsT[index]
                    + globalTransformsT[fk.getJointParent(index)];

        }

        for(int i=0; i<numIKJoints; i++){
            int index = IKJointIDs[i];
            Vec3<real> handle = {0,0,0};
            handle = globalTransformsR[index] * handle + globalTransformsT[index];
            handlePositions[i*3+0] = handle[0];
            handlePositions[i*3+1] = handle[1];
            handlePositions[i*3+2] = handle[2];
        }

}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  // Students should implement this.
  // Here, you should setup adol_c:
  //   Define adol_c inputs and outputs. 
  //   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
  //   This will later make it possible for you to compute the gradient of this function in IK::doIK
  //   (in other words, compute the "Jacobian matrix" J).
  // See ADOLCExample.cpp .

    trace_on(adolc_tagID); // start tracking computation with ADOL-C
    vector<adouble> x(FKInputDim); // define the input of the function f
    for(int i = 0; i < FKInputDim; i++)
        x[i] <<= 0.0; // The <<= syntax tells ADOL-C that these are the input variables.

    vector<adouble> y(FKOutputDim); // define the output of the function f

    forwardKinematicsFunction(numIKJoints,IKJointIDs,*fk, x,y);

    vector<double> output(FKOutputDim);
    for(int i = 0; i < FKOutputDim; i++)
        y[i] >>= output[i]; // Use >>= to tell ADOL-C that y[i] are the output variables

    // Finally, call trace_off to stop recording the function f.
    trace_off(); // ADOL-C tracking finished
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
  // You may find the following helpful:
  int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!


    double input[FKInputDim];
    for(int i=0; i<numJoints; i++){
        input[i*3+0] = jointEulerAngles[i][0];
        input[i*3+1] = jointEulerAngles[i][1];
        input[i*3+2] = jointEulerAngles[i][2];
    }
//    double output[FKOutputDim];
//    for(int i=0; i<FKOutputDim; i++) output[i] = 0;
//    ::function(adolc_tagID, FKOutputDim, FKInputDim, input, output);

    Eigen::VectorXd dx(FKOutputDim);
    for(int i=0; i<numIKJoints; i++){
//        dx(i*3+0) = targetHandlePositions[i][0] - output[i*3+0];
//        dx(i*3+1) = targetHandlePositions[i][1] - output[i*3+1];
//        dx(i*3+2) = targetHandlePositions[i][2] - output[i*3+2];
        dx(i*3+0) = targetHandlePositions[i][0] - fk->getJointGlobalPosition(IKJointIDs[i])[0];
        dx(i*3+1) = targetHandlePositions[i][1] - fk->getJointGlobalPosition(IKJointIDs[i])[1];
        dx(i*3+2) = targetHandlePositions[i][2] - fk->getJointGlobalPosition(IKJointIDs[i])[2];
    }

    double jacobianMatrix[FKOutputDim*FKInputDim]; // We store the matrix in row-major order.
    double * jacobianMatrixEachRow[FKOutputDim];
    for(int i=0; i<FKOutputDim; i++)
        jacobianMatrixEachRow[i] = &jacobianMatrix[i*FKInputDim];
    ::jacobian(adolc_tagID, FKOutputDim, FKInputDim, input, jacobianMatrixEachRow); // each row is the gradient of one output component of the function

    Eigen::MatrixXd J(FKOutputDim,FKInputDim);
    for(int i=0; i<FKOutputDim; i++)
        for(int j=0; j<FKInputDim; j++)
            J(i,j) = jacobianMatrix[i*FKInputDim+j];


    Eigen::MatrixXd Jt = J.transpose();

    Eigen::VectorXd rhs = Jt * dx;
    Eigen::MatrixXd mu = 0.01*Eigen::MatrixXd::Identity(FKInputDim,FKInputDim);
    Eigen::MatrixXd A = Jt * J + mu;
    Eigen::VectorXd dTheta = A.ldlt().solve(rhs);

//    Eigen::VectorXd dTheta = Jt* ((J * Jt).inverse()) * dx;

    for(int i=0; i<numJoints; i++){
        jointEulerAngles[i][0] += dTheta(i*3+0);
        jointEulerAngles[i][1] += dTheta(i*3+1);
        jointEulerAngles[i][2] += dTheta(i*3+2);
    }
//    jointEulerAngles[14]+=Vec3d(0.5, 0, 0);;
    // Students should implement this.
  // Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
  // Specifically, use ::function, and ::jacobian .
  // See ADOLCExample.cpp .
  //
  // Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
  // Note that at entry, "jointEulerAngles" contains the input Euler angles. 
  // Upon exit, jointEulerAngles should contain the new Euler angles.
}

