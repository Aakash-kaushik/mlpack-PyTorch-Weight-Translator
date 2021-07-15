/**
 * @file imagenette_trainer.hpp
 * @author Aakash Kaushik
 *
 * Contains implementation of object classification suite. It can be used
 * to select object classification model, it's parameter dataset and
 * other training parameters.
 *
 * NOTE: This code needs to be adapted as this implementation doesn't support
 *       Command Line Arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <map>
#include <mlpack/core.hpp>
#include <dataloader/dataloader.hpp>
#include <models/mobilenet/mobilenet_v1.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
#include <sstream>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::models;
using namespace arma;
using namespace std;
using namespace ens;

std::queue<std::string> batchNormRunningMean;
std::queue<std::string> batchNormRunningVar;

class Accuracy
{
 public:
  template<typename InputType, typename OutputType>
  static double Evaluate(InputType& input, OutputType& output)
  {
    arma::Row<size_t> predLabels(input.n_cols);
    for (arma::uword i = 0; i < input.n_cols; ++i)
    {
      predLabels(i) = input.col(i).index_max() + 1;
    }
    return arma::accu(predLabels == output) / (double)output.n_elem * 100;
  }
};


template <
    typename OutputLayer = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void LoadWeights(mlpack::ann::FFN<OutputLayer, InitializationRule> &model,
                 std::string modelConfigPath)
{
  std::cout << "Loading Weights\n";
  size_t currentOffset = 0;
  boost::property_tree::ptree xmlFile;
  boost::property_tree::read_xml(modelConfigPath, xmlFile);
  boost::property_tree::ptree modelConfig = xmlFile.get_child("model");

  model.Parameters().fill(0);
  BOOST_FOREACH (boost::property_tree::ptree::value_type const &layer, modelConfig)
  {
    std::string progressBar(81, '-');
    size_t filled = std::ceil(currentOffset * 80.0 / model.Parameters().n_elem);
    progressBar[0] = '[';
    std::fill(progressBar.begin() + 1, progressBar.begin() + filled + 1, '=');
    std::cout << progressBar << "] " << filled * 100.0 / 80.0 << "%\r";
    std::cout.flush();

    // Load Weights.
    if (layer.second.get_child("has_weights").data() != "0")
    { 
      arma::mat weights;
      mlpack::data::Load("./../../" + layer.second.get_child("weight_csv").data(), weights);
      model.Parameters()(arma::span(currentOffset, currentOffset + weights.n_elem - 1),
                         arma::span()) = weights.t();
      currentOffset += weights.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("weight_offset").data());
    }

    // Load Biases.
    if (layer.second.get_child("has_bias").data() != "0")
    {
      arma::mat bias;
      mlpack::data::Load("./../../" + layer.second.get_child("bias_csv").data(), bias);
      model.Parameters()(arma::span(currentOffset, currentOffset + bias.n_elem - 1),
                         arma::span()) = bias.t();
      currentOffset += bias.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("bias_offset").data());
    }

    if (layer.second.get_child("has_running_mean").data() != "0")
    {
      batchNormRunningMean.push("./../../" + layer.second.get_child("running_mean_csv").data());
    }

    if (layer.second.get_child("has_running_var").data() != "0")
    {
      batchNormRunningVar.push("./../../" + layer.second.get_child("running_var_csv").data());
    }
  }
  std::cout << std::endl;
  std::cout << "Loaded Weights\n";
}

void LoadBNMats(arma::mat& runningMean, arma::mat& runningVar)
{
  runningMean.clear();
  if (!batchNormRunningMean.empty())
  {
    cout << batchNormRunningMean.front() << endl;
    mlpack::data::Load(batchNormRunningMean.front(), runningMean);
    batchNormRunningMean.pop();
  }
  else
    std::cout << "This should never happen!\n";

  runningVar.clear();
  if (!batchNormRunningVar.empty())
  {
    cout << batchNormRunningVar.front() << endl;
    mlpack::data::Load(batchNormRunningVar.front(), runningVar);
    batchNormRunningVar.pop();
  }
  else
    std::cout << "This should never happen!\n";
}

template <
    typename OutputLayer = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRule = mlpack::ann::RandomInitialization
>void HardCodedRunningMeanAndVariance(
    mlpack::ann::FFN<OutputLayer, InitializationRule>& model)
{
  arma::mat runningMean, runningVar;
  vector<size_t> indices ={ 1 };
  for (size_t idx : indices)
  {
    LoadBNMats(runningMean, runningVar);
    std::cout << "Loading RunningMean and Variance for " << idx << std::endl;
    boost::get<BatchNorm<>*>(model.Model()[idx])->TrainingMean() = runningMean.t();
    boost::get<BatchNorm<>*>(model.Model()[idx])->TrainingVariance() = runningVar.t();
    boost::get<BatchNorm<>*>(model.Model()[idx])->Deterministic() = true;
  }

  std::map<size_t, size_t> mobilenetcfg = {
    {3, 0}, {4, 1}, {5, 0}, {6, 1}, {7, 0}, {8, 1}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 1}, {15, 0}
  };

  for (auto it = mobilenetcfg.begin(); it != mobilenetcfg.end(); ++it)
  {
    if (it->second == 0)
    {
      LoadBNMats(runningMean, runningVar);
      std::cout << "Loading RunningMean and Variance for non padding blocks : " << it->first << std::endl;

      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[1])->Deterministic() = true;

      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[4])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[4])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[4])->Deterministic() = true;
    }
    if  (it->second == 1)
    {
      LoadBNMats(runningMean, runningVar);
      std::cout << "Loading RunningMean and Variance for padding blocks : " << it->first << std::endl;

      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[2])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[2])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[2])->Deterministic() = true;

      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[5])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[5])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[5])->Deterministic() = true;      
    }
  }
}


int main()
{ 
  std::vector<double> alpha = {0.25, 0.5, 0.75, 1.0};
  std::vector<int> image_size = {128, 160, 192, 224};
  for (double alpha_val : alpha)
  {
    for (int image_size_val : image_size)
    {
      MobilenetV1 mobilenet(3, image_size_val, image_size_val, alpha_val);
      std::ostringstream alp, img_sz;
      alp << alpha_val;
      img_sz << image_size_val;
      string config_file = "mobilenet_v1_size_" + img_sz.str() + "_alpha_" +  alp.str() + ".xml";
      string save_file = "mobilenetv1_" + alp.str() + "_" + img_sz.str() + ".xml";
      std::cout << config_file << std::endl;
      LoadWeights<mlpack::ann::CrossEntropyError<> >(mobilenet.GetModel(), "./../../cfg/" + config_file);
      HardCodedRunningMeanAndVariance<mlpack::ann::CrossEntropyError<> >(mobilenet.GetModel());
      mlpack::data::Save("./weights/"+save_file+".bin", "mobilenet_v1", mobilenet.GetModel());
    }
  } 

  // arma::mat input(224 * 224 * 3, 1), output;
  // input.fill(1.0);
  // mobilenet.GetModel().Predict(input, output);
  // double sum = arma::accu(output);
  // std::cout << sum << std::endl;
  // output.print();

  // std::cout << "Loaded model output" << std::endl;
  // mobilenet2.LoadModel("./mobilenet_1.0_224.bin");
  // mobilenet2.GetModel().Predict(input, output);
  // sum = arma::accu(output);
  // std::cout << sum << std::endl;
  // output.print();

  // mlpack::data::Load("./../../../../imagenette_image.csv", input);
  // std::cout << input.n_cols << std::endl;
  // if (input.n_cols > 80)
  // {
  //     input = input.t();
  //     cout << "New cols : " << input.n_cols << std::endl;
  // }

  // for (int i = 0; i < input.n_cols; i++)
  // {
  //     output.clear();
  //     resnet.GetModel().Predict(input.col(i), output);
  //     sum = arma::accu(output);
  //     std::cout << std::setprecision(10) << sum << " --> " << output.col(0).index_max() <<
  //         "  " << output.col(0).max() << std::endl;
  // }
  return 0;
}