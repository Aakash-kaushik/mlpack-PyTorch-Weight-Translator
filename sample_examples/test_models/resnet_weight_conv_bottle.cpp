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
#include <models/resnet/resnet.hpp>
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
    boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[0])->Model()[1])->TrainingMean() = runningMean.t();
    boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[0])->Model()[1])->TrainingVariance() = runningVar.t();
    boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[0])->Model()[1])->Deterministic() = true;
  }

  std::map<size_t, size_t> resnetcfg = {
    {4, 0},
    {5, 1},
    {6, 1},
    {7, 0},
    {8, 1},
    {9, 1},
    {10, 1},
    {11, 0},
    {12, 1},
    {13, 1},
    {14, 1},
    {15, 1},
    {16, 1},
    {17, 0},
    {18, 1},
    {19, 1}
 };

  for (auto it = resnetcfg.begin(); it != resnetcfg.end(); ++it)
  {
    // for identitiy blocks
    if (it->second == 1)
    {
      std::cout << "Loading RunningMean and Variance for identitiy blocks: " << it->first << std::endl;
      LoadBNMats(runningMean, runningVar);

      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[0])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[0])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[0])->Model()[1])->Deterministic() = true;

      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[2])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[2])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[2])->Model()[1])->Deterministic() = true;

      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[4])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[4])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[4])->Model()[1])->Deterministic() = true;
    }

    // For downsample blocks. 
    else if (it->second == 0)
    {
      std::cout << "Loading RunningMean and Variance for downsample blocks: " << it->first << std::endl;
      LoadBNMats(runningMean, runningVar);

      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[0])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[0])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[0])->Model()[1])->Deterministic() = true;

      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[2])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[2])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[2])->Model()[1])->Deterministic() = true;

      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[4])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[4])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[0])->Model()[4])->Model()[1])->Deterministic() = true;
      
      LoadBNMats(runningMean, runningVar);
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[1])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[1])->Model()[1])->TrainingVariance() = runningVar.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<AddMerge<>*>(boost::get<Sequential<>*>(model.Model()[it->first])->Model()[0])->Model()[1])->Model()[1])->Deterministic() = true;
    }
  }
}


int main()
{ 
  ResNet50 resnet(3, 224, 224);
  LoadWeights<mlpack::ann::CrossEntropyError<> >(resnet.GetModel(), "./../../cfg/resnet50.xml");
  HardCodedRunningMeanAndVariance<mlpack::ann::CrossEntropyError<> >(resnet.GetModel());

  arma::mat input(224 * 224 * 3, 1), output;
  input.fill(1.0);
  resnet.GetModel().Predict(input, output);
  double sum = arma::accu(output);
  std::cout << sum << std::endl;
  mlpack::data::Save("./weights/resnet50.bin", "ResNet", resnet.GetModel());

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