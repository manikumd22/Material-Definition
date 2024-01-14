#pragma once

#include "ComputeLagrangianStressPK2.h"
#include "DenseMatrix.h"

class ComputeNeoHookeanStress : public ComputeLagrangianStressPK2
{
public:
  static InputParameters validParams();
  ComputeNeoHookeanStress(const InputParameters & parameters);

protected:
  /// Actual stress/Jacobian update
  virtual void computeQpPK2Stress();

protected:
  const MaterialProperty<Real> & _lambda;
  const MaterialProperty<Real> & _mu;
  MaterialProperty<DenseMatrix<Real>> & _F_column;
  const std::vector<Real> _fc1_weights;
  MaterialProperty<DenseMatrix<Real>> & _weight_layer_1;
  const std::vector<Real> _fc1_bias;
  MaterialProperty<DenseMatrix<Real>> & _weight_bias_1;
  MaterialProperty<DenseMatrix<Real>> & _weight_output_layer_1;
  const std::vector<Real> _bn1_weights;
  const std::vector<Real> _bn1_bias;
  const std::vector<Real> _bn1_running_mean;
  const std::vector<Real> _bn1_running_variance;
  MaterialProperty<DenseMatrix<Real>> & _running_mean_bn1;
  MaterialProperty<DenseMatrix<Real>> & _running_variance_bn1;
  MaterialProperty<DenseMatrix<Real>> & _weights_bn1;
  MaterialProperty<DenseMatrix<Real>> & _bias_bn1;
  MaterialProperty<DenseMatrix<Real>> & _weight_input_layer_2;
  const std::vector<Real> _fc2_weights;
  MaterialProperty<DenseMatrix<Real>> & _weight_layer_2;
  const std::vector<Real> _fc2_bias;
  MaterialProperty<DenseMatrix<Real>> & _weight_bias_2;
  MaterialProperty<DenseMatrix<Real>> & _weight_output_layer_2;
  const std::vector<Real> _bn2_weights;
  const std::vector<Real> _bn2_bias;
  const std::vector<Real> _bn2_running_mean;
  const std::vector<Real> _bn2_running_variance;
  MaterialProperty<DenseMatrix<Real>> & _running_mean_bn2;
  MaterialProperty<DenseMatrix<Real>> & _running_variance_bn2;
  MaterialProperty<DenseMatrix<Real>> & _weights_bn2;
  MaterialProperty<DenseMatrix<Real>> & _bias_bn2;  
  MaterialProperty<DenseMatrix<Real>> & _weight_input_layer_3;
  const std::vector<Real> _fc3_weights;
  MaterialProperty<DenseMatrix<Real>> & _weight_layer_3;
  const std::vector<Real> _fc3_bias;
  MaterialProperty<DenseMatrix<Real>> & _weight_bias_3;
  MaterialProperty<DenseMatrix<Real>> & _weight_output_layer_3;
  const std::vector<Real> _bn3_weights;
  const std::vector<Real> _bn3_bias;
  const std::vector<Real> _bn3_running_mean;
  const std::vector<Real> _bn3_running_variance;
  MaterialProperty<DenseMatrix<Real>> & _running_mean_bn3;
  MaterialProperty<DenseMatrix<Real>> & _running_variance_bn3;
  MaterialProperty<DenseMatrix<Real>> & _weights_bn3;
  MaterialProperty<DenseMatrix<Real>> & _bias_bn3;  
  MaterialProperty<DenseMatrix<Real>> & _weight_input_layer_4;
  const std::vector<Real> _fc4_weights;
  MaterialProperty<DenseMatrix<Real>> & _weight_layer_4;
  const std::vector<Real> _fc4_bias;
  MaterialProperty<DenseMatrix<Real>> & _weight_bias_4;
  MaterialProperty<DenseMatrix<Real>> & _weight_output_layer_4;
  const std::vector<Real> _bn4_weights;
  const std::vector<Real> _bn4_bias;
  const std::vector<Real> _bn4_running_mean;
  const std::vector<Real> _bn4_running_variance;
  MaterialProperty<DenseMatrix<Real>> & _running_mean_bn4;
  MaterialProperty<DenseMatrix<Real>> & _running_variance_bn4;
  MaterialProperty<DenseMatrix<Real>> & _weights_bn4;
  MaterialProperty<DenseMatrix<Real>> & _bias_bn4;  
  MaterialProperty<DenseMatrix<Real>> & _weight_input_layer_5;
  const std::vector<Real> _fc5_weights;
  MaterialProperty<DenseMatrix<Real>> & _weight_layer_5;
  const std::vector<Real> _fc5_bias;
  MaterialProperty<DenseMatrix<Real>> & _weight_bias_5;
  MaterialProperty<DenseMatrix<Real>> & _weight_output_layer_5;
  MaterialProperty<DenseMatrix<Real>> & _weight_input_layer_6;
  // MaterialProperty<std::vector<Real>> & _weight_constants_material;
};