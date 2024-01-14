#pragma once

#include "ComputeLagrangianStressPK2.h"
#include "DenseMatrix.h"

class ComputeFractureStress : public ComputeLagrangianStressPK2
{
public:
  static InputParameters validParams();
  ComputeFractureStress(const InputParameters & parameters);

protected:
  /// Actual stress/Jacobian update
  virtual void computeQpPK2Stress();

protected:
  const MaterialProperty<Real> & _lambda;
  const MaterialProperty<Real> & _mu;
};
