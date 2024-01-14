#include "ComputeNeoHookeanStress.h"

registerMooseObject("TensorMechanicsApp", ComputeNeoHookeanStress);

InputParameters
ComputeNeoHookeanStress::validParams()
{
  InputParameters params = ComputeLagrangianStressPK2::validParams();
  params.addParam<MaterialPropertyName>("lambda",
                                        "lambda",
                                        "Parameter conjugate to Lame parameter"
                                        " for small deformations");
  params.addParam<MaterialPropertyName>("mu",
                                        "mu",
                                        "Parameter conjugate to Lame parameter"
                                        " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc1_weights",
                                                         "fc1_weights_size = 256",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc1_bias",
                                                         "fc1_bias_size = 64",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn1_weights",
                                                         "bn1_weights_size = 64",
                                                         " for small deformations"); 
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn1_bias",
                                                         "bn1_bias_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn1_running_mean",
                                                         "bn1_running_mean_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn1_running_variance",
                                                         "bn1_running_variance_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc2_weights",
                                                         "fc2_weights_size = 4096",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc2_bias",
                                                         "fc2_bias_size = 64",
                                                         " for small deformations"); 
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn2_weights",
                                                         "bn2_weights_size = 64",
                                                         " for small deformations"); 
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn2_bias",
                                                         "bn2_bias_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn2_running_mean",
                                                         "bn2_running_mean_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn2_running_variance",
                                                         "bn2_running_variance_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc3_weights",
                                                         "fc3_weights_size = 4096",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc3_bias",
                                                         "fc3_bias_size = 64",
                                                         " for small deformations");  
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn3_weights",
                                                         "bn3_weights_size = 64",
                                                         " for small deformations"); 
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn3_bias",
                                                         "bn3_bias_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn3_running_mean",
                                                         "bn3_running_mean_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn3_running_variance",
                                                         "bn3_running_variance_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc4_weights",
                                                         "fc4_weights_size = 4096",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc4_bias",
                                                         "fc4_bias_size = 64",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn4_weights",
                                                         "bn4_weights_size = 64",
                                                         " for small deformations"); 
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn4_bias",
                                                         "bn4_bias_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn4_running_mean",
                                                         "bn4_running_mean_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("bn4_running_variance",
                                                         "bn4_running_variance_size = 64",
                                                         " for small deformations");      
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc5_weights",
                                                         "fc5_weights_size = 192",
                                                         " for small deformations");
  params.addRequiredRangeCheckedParam<std::vector<Real>>("fc5_bias",
                                                         "fc5_bias_size = 3",
                                                         " for small deformations");
  return params;
}
ComputeNeoHookeanStress::ComputeNeoHookeanStress(const InputParameters & parameters)
  : ComputeLagrangianStressPK2(parameters),
    _lambda(getMaterialProperty<Real>(getParam<MaterialPropertyName>("lambda"))),
    _mu(getMaterialProperty<Real>(getParam<MaterialPropertyName>("mu"))),
    _F_column(declareProperty<DenseMatrix<Real>>("F_column")),  
    _fc1_weights(getParam<std::vector<Real>>("fc1_weights")),
    _weight_layer_1(declareProperty<DenseMatrix<Real>>("weight_layer_1")),
    _fc1_bias(getParam<std::vector<Real>>("fc1_bias")),
    _weight_bias_1(declareProperty<DenseMatrix<Real>>("weight_bias_1")),
    _weight_output_layer_1(declareProperty<DenseMatrix<Real>>("weight_output_layer_1")),
    _bn1_weights(getParam<std::vector<Real>>("bn1_weights")),
    _bn1_bias(getParam<std::vector<Real>>("bn1_bias")),
    _bn1_running_mean(getParam<std::vector<Real>>("bn1_running_mean")),
    _bn1_running_variance(getParam<std::vector<Real>>("bn1_running_variance")),
    _running_mean_bn1(declareProperty<DenseMatrix<Real>>("running_mean_bn1")),
    _running_variance_bn1(declareProperty<DenseMatrix<Real>>("running_variance_bn1")),
    _weights_bn1(declareProperty<DenseMatrix<Real>>("running_mean_bn1")),
    _bias_bn1(declareProperty<DenseMatrix<Real>>("running_variance_bn1")),    
    _weight_input_layer_2(declareProperty<DenseMatrix<Real>>("weight_input_layer_2")),
    _fc2_weights(getParam<std::vector<Real>>("fc2_weights")),
    _weight_layer_2(declareProperty<DenseMatrix<Real>>("weight_layer_2")),
    _fc2_bias(getParam<std::vector<Real>>("fc2_bias")),
    _weight_bias_2(declareProperty<DenseMatrix<Real>>("weight_bias_2")),
    _weight_output_layer_2(declareProperty<DenseMatrix<Real>>("weight_output_layer_2")),
    _bn2_weights(getParam<std::vector<Real>>("bn2_weights")),
    _bn2_bias(getParam<std::vector<Real>>("bn2_bias")),
    _bn2_running_mean(getParam<std::vector<Real>>("bn2_running_mean")),
    _bn2_running_variance(getParam<std::vector<Real>>("bn2_running_variance")),  
    _running_mean_bn2(declareProperty<DenseMatrix<Real>>("running_mean_bn2")),
    _running_variance_bn2(declareProperty<DenseMatrix<Real>>("running_variance_bn2")),
    _weights_bn2(declareProperty<DenseMatrix<Real>>("weights_bn2")),
    _bias_bn2(declareProperty<DenseMatrix<Real>>("bias_bn2")),        
    _weight_input_layer_3(declareProperty<DenseMatrix<Real>>("weight_input_layer_3")),
    _fc3_weights(getParam<std::vector<Real>>("fc3_weights")),
    _weight_layer_3(declareProperty<DenseMatrix<Real>>("weight_layer_3")),
    _fc3_bias(getParam<std::vector<Real>>("fc3_bias")),
    _weight_bias_3(declareProperty<DenseMatrix<Real>>("weight_bias_3")),
    _weight_output_layer_3(declareProperty<DenseMatrix<Real>>("weight_output_layer_3")),
    _bn3_weights(getParam<std::vector<Real>>("bn3_weights")),
    _bn3_bias(getParam<std::vector<Real>>("bn3_bias")),
    _bn3_running_mean(getParam<std::vector<Real>>("bn3_running_mean")),
    _bn3_running_variance(getParam<std::vector<Real>>("bn3_running_variance")), 
    _running_mean_bn3(declareProperty<DenseMatrix<Real>>("running_mean_bn3")),
    _running_variance_bn3(declareProperty<DenseMatrix<Real>>("running_variance_bn3")),
    _weights_bn3(declareProperty<DenseMatrix<Real>>("weights_bn3")),
    _bias_bn3(declareProperty<DenseMatrix<Real>>("bias_bn3")),        
    _weight_input_layer_4(declareProperty<DenseMatrix<Real>>("weight_input_layer_4")),
    _fc4_weights(getParam<std::vector<Real>>("fc4_weights")),
    _weight_layer_4(declareProperty<DenseMatrix<Real>>("weight_layer_4")),
    _fc4_bias(getParam<std::vector<Real>>("fc4_bias")),
    _weight_bias_4(declareProperty<DenseMatrix<Real>>("weight_bias_4")),
    _weight_output_layer_4(declareProperty<DenseMatrix<Real>>("weight_output_layer_4")),
    _bn4_weights(getParam<std::vector<Real>>("bn4_weights")),
    _bn4_bias(getParam<std::vector<Real>>("bn4_bias")),
    _bn4_running_mean(getParam<std::vector<Real>>("bn4_running_mean")),
    _bn4_running_variance(getParam<std::vector<Real>>("bn4_running_variance")),
    _running_mean_bn4(declareProperty<DenseMatrix<Real>>("running_mean_bn4")),
    _running_variance_bn4(declareProperty<DenseMatrix<Real>>("running_variance_bn4")),
    _weights_bn4(declareProperty<DenseMatrix<Real>>("weights_bn4")),
    _bias_bn4(declareProperty<DenseMatrix<Real>>("bias_bn4")),  
    _weight_input_layer_5(declareProperty<DenseMatrix<Real>>("weight_input_layer_5")),
    _fc5_weights(getParam<std::vector<Real>>("fc5_weights")),
    _weight_layer_5(declareProperty<DenseMatrix<Real>>("weight_layer_5")),
    _fc5_bias(getParam<std::vector<Real>>("fc5_bias")),
    _weight_bias_5(declareProperty<DenseMatrix<Real>>("weight_bias_5")),
    _weight_output_layer_5(declareProperty<DenseMatrix<Real>>("weight_output_layer_5")),    
    _weight_input_layer_6(declareProperty<DenseMatrix<Real>>("weight_input_layer_6"))
{
}

void
ComputeNeoHookeanStress::computeQpPK2Stress()
{
  // Hyperelasticity is weird, we need to branch on the type of update if we
  // want a truly linear model
  //
  // This is because we need to drop quadratic terms for the linear update
  usingTensorIndices(i_, j_, k_, l_);

  // Large deformation = nonlinear strain
  if (_large_kinematics)
  {
    RankTwoTensor Cinv = (2 * _E[_qp] + RankTwoTensor::Identity()).inverse();
    _S[_qp] = (_lambda[_qp] * log(_F[_qp].det()) - _mu[_qp]) * Cinv +
              _mu[_qp] * RankTwoTensor::Identity();
    _C[_qp] =
        -2 * (_lambda[_qp] * log(_F[_qp].det()) - _mu[_qp]) * Cinv.times<i_, k_, l_, j_>(Cinv) +
        _lambda[_qp] * Cinv.times<i_, j_, k_, l_>(Cinv);
  }
  // Small deformations = linear strain
  else
  {
    const auto I = RankTwoTensor::Identity();
    _weight_layer_1[_qp].resize(64, 4);
    int index1 = 0;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 4; ++j) {
            _weight_layer_1[_qp](i, j) = _fc1_weights[index1];
            ++index1;
        }
    }   

    int index2 = 0;
    _weight_layer_2[_qp].resize(64, 64);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            _weight_layer_2[_qp](i, j) = _fc2_weights[index2];
            ++index2;
        }
    }

    int index3 = 0;
    _weight_layer_3[_qp].resize(64, 64);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            _weight_layer_3[_qp](i, j) = _fc3_weights[index3];
            ++index3;
        }
    }  

    int index4 = 0;
    _weight_layer_4[_qp].resize(64, 64);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            _weight_layer_4[_qp](i, j) = _fc4_weights[index3];
            ++index4;
        }
    }  

    int index5 = 0;
    _weight_layer_5[_qp].resize(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            _weight_layer_5[_qp](i, j) = _fc5_weights[index5];
            ++index5;
        }
    }   

    _weight_bias_1[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_bias_1[_qp](i, 0) = _fc1_bias[i];
    }
    _weight_bias_2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_bias_2[_qp](i, 0) = _fc2_bias[i];
    }
    _weight_bias_3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_bias_3[_qp](i, 0) = _fc3_bias[i];
    }
    _weight_bias_4[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_bias_4[_qp](i, 0) = _fc4_bias[i];
    }
    _weight_bias_5[_qp].resize(3, 1);
    for (int i = 0; i < 3; ++i) {
        _weight_bias_5[_qp](i, 0) = _fc5_bias[i];
    }
    _running_mean_bn1[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_mean_bn1[_qp](i, 0) = _bn1_running_mean[i];
    }
    _running_variance_bn1[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_variance_bn1[_qp](i, 0) = _bn1_running_variance[i];
    }
    _weights_bn1[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weights_bn1[_qp](i, 0) = _bn1_weights[i];
    }
    _bias_bn1[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _bias_bn1[_qp](i, 0) = _bn1_bias[i];
    }
    _running_mean_bn2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_mean_bn2[_qp](i, 0) = _bn2_running_mean[i];
    }
    _running_variance_bn2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_variance_bn2[_qp](i, 0) = _bn2_running_variance[i];
    }
    _weights_bn2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weights_bn2[_qp](i, 0) = _bn2_weights[i];
    }
    _bias_bn2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _bias_bn2[_qp](i, 0) = _bn2_bias[i];
    }
    _running_mean_bn3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_mean_bn3[_qp](i, 0) = _bn3_running_mean[i];
    }
    _running_variance_bn3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_variance_bn1[_qp](i, 0) = _bn3_running_variance[i];
    }
    _weights_bn3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weights_bn3[_qp](i, 0) = _bn3_weights[i];
    }
    _bias_bn3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _bias_bn3[_qp](i, 0) = _bn3_bias[i];
    }

    _running_mean_bn4[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_mean_bn4[_qp](i, 0) = _bn4_running_mean[i];
    }
    _running_variance_bn4[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _running_variance_bn4[_qp](i, 0) = _bn4_running_variance[i];
    }
    _weights_bn4[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weights_bn4[_qp](i, 0) = _bn4_weights[i];
    }
    _bias_bn4[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _bias_bn4[_qp](i, 0) = _bn4_bias[i];
    }


    _F_column[_qp].resize(4, 1);
    _F_column[_qp](0,0) = (_F[_qp](0,0) - 1.00000000e+00)/(0.2312286 + 1e-5);
    _F_column[_qp](1,0) = (_F[_qp](0,1) + 3.45655972e-19 )/(0.0233809 + 1e-5);
    _F_column[_qp](2,0) = (_F[_qp](1,0) + 3.45655972e-19)/(0.0233809 + 1e-5);
    _F_column[_qp](3,0) = (_F[_qp](1,1) - 1.00000000e+00)/(0.2312286 + 1e-5);

    _weight_output_layer_1[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_output_layer_1[_qp](i, 0) = 0.0;
        for (int j = 0; j < 4; ++j) {
            _weight_output_layer_1[_qp](i, 0) += _weight_layer_1[_qp](i, j) * _F_column[_qp](j,0);
        }
    }    
    for (int i = 0; i < 64; ++i){
        _weight_output_layer_1[_qp](i, 0) = _weight_output_layer_1[_qp](i, 0) + _weight_bias_1[_qp](i, 0);
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_1[_qp](i, 0) =  (_weight_output_layer_1[_qp](i, 0) - _running_mean_bn1[_qp](i,0))/(_running_variance_bn1[_qp](i,0) + 1e-5);   
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_1[_qp](i, 0) = _weights_bn1[_qp](i, 0) * _weight_output_layer_1[_qp](i, 0) + _bias_bn1[_qp](i, 0);
    }


    _weight_input_layer_2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        Real value = _weight_output_layer_1[_qp](i, 0);
        _weight_input_layer_2[_qp](i, 0) = std::max(value, 0.0);
    }

    _weight_output_layer_2[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_output_layer_2[_qp](i, 0) = 0.0;
        for (int j = 0; j < 64; ++j) {
            _weight_output_layer_2[_qp](i, 0) += _weight_layer_2[_qp](i, j) * _weight_input_layer_2[_qp](j, 0);
        }
        _weight_output_layer_2[_qp](i, 0) = _weight_output_layer_2[_qp](i, 0) + _weight_bias_2[_qp](i, 0);
    }    

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_2[_qp](i, 0) =  (_weight_output_layer_2[_qp](i, 0) - _running_mean_bn2[_qp](i,0))/(_running_variance_bn2[_qp](i,0) + 1e-5);   
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_2[_qp](i, 0) = _weights_bn2[_qp](i, 0) * _weight_output_layer_2[_qp](i, 0) + _bias_bn2[_qp](i, 0);
    }

    _weight_input_layer_3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        Real value = _weight_output_layer_2[_qp](i, 0);
        _weight_input_layer_3[_qp](i, 0) = std::max(value, 0.0); 
    }

    _weight_output_layer_3[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_output_layer_3[_qp](i, 0) = 0.0;
        for (int j = 0; j < 64; ++j) {
            _weight_output_layer_3[_qp](i, 0) += _weight_layer_3[_qp](i, j) * _weight_input_layer_3[_qp](j, 0);
        }
        _weight_output_layer_3[_qp](i, 0) = _weight_output_layer_3[_qp](i, 0) + _weight_bias_3[_qp](i, 0);
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_3[_qp](i, 0) =  (_weight_output_layer_3[_qp](i, 0) - _running_mean_bn3[_qp](i,0))/(_running_variance_bn3[_qp](i,0) + 1e-5);   
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_3[_qp](i, 0) = _weights_bn3[_qp](i, 0) * _weight_output_layer_3[_qp](i, 0) + _bias_bn3[_qp](i, 0);
    }


    _weight_input_layer_4[_qp].resize(128, 1);

    for (int i = 0; i < 64; ++i) {
        Real value = _weight_output_layer_3[_qp](i, 0);
        _weight_input_layer_4[_qp](i, 0) = std::max(value, 0.0);
    }

    _weight_output_layer_4[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        _weight_output_layer_4[_qp](i, 0) = 0.0;
        for (int j = 0; j < 64; ++j) {
            _weight_output_layer_4[_qp](i, 0) += _weight_layer_4[_qp](i, j) * _weight_input_layer_4[_qp](j, 0);
        }
        _weight_output_layer_4[_qp](i, 0) = _weight_output_layer_4[_qp](i, 0) + _weight_bias_4[_qp](i, 0);
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_4[_qp](i, 0) =  (_weight_output_layer_4[_qp](i, 0) - _running_mean_bn4[_qp](i,0))/(_running_variance_bn4[_qp](i,0) + 1e-5);   
    }

    for (int i = 0; i < 64; ++i){
        _weight_output_layer_4[_qp](i, 0) = _weights_bn4[_qp](i, 0) * _weight_output_layer_4[_qp](i, 0) + _bias_bn4[_qp](i, 0);
    }

    _weight_input_layer_5[_qp].resize(64, 1);
    for (int i = 0; i < 64; ++i) {
        Real value = _weight_output_layer_4[_qp](i, 0);
        _weight_input_layer_5[_qp](i, 0) = std::max(value, 0.0); 
    }  

    _weight_output_layer_5[_qp].resize(3, 1);
    for (int i = 0; i < 3; ++i) {
        _weight_output_layer_5[_qp](i, 0) = 0.0;
        for (int j = 0; j < 64; ++j) {
            _weight_output_layer_5[_qp](i, 0) += _weight_layer_5[_qp](i, j) * _weight_input_layer_5[_qp](j, 0);
        }
        _weight_output_layer_5[_qp](i, 0) = _weight_output_layer_5[_qp](i, 0) + _weight_bias_5[_qp](i, 0);
    }

    RankTwoTensor strain = 0.5 * (_F[_qp] + _F[_qp].transpose()) - I;
    _C[_qp] = _lambda[_qp] * I.times<i_, j_, k_, l_>(I) +
            2.0 * _mu[_qp] * RankFourTensor(RankFourTensor::initIdentitySymmetricFour);
    _S[_qp] = _C[_qp] * strain;

    _weight_output_layer_5[_qp](0,0) = _weight_output_layer_5[_qp](0,0)*(0.0057805 + 1e-5) + 2.49990626e-02;
    _weight_output_layer_5[_qp](1,0) = _weight_output_layer_5[_qp](1,0)*(0.0005845 + 1e-5) - 1.93347871e-11;
    _weight_output_layer_5[_qp](2,0) = _weight_output_layer_5[_qp](2,0)*(0.0057805 + 1e-5) + 2.49990626e-02;

    _S[_qp](0,0) = 10000*_weight_output_layer_5[_qp](0,0);
    _S[_qp](0,1) = 10000*_weight_output_layer_5[_qp](1,0);
    _S[_qp](1,0) = 10000*_weight_output_layer_5[_qp](1,0);
    _S[_qp](1,1) = 10000*_weight_output_layer_5[_qp](2,0);
  }
}

