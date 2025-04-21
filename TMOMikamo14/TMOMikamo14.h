/*******************************************************************************
 *                                                                              *
 *                         Brno University of Technology                        *
 *                       Faculty of Information Technology                      *
 *                                                                              *
 *                 A tone reproduction operator for all luminance               *
 *                   ranges considering human color perception                  *
 * 																			                                        *
 *                                 Bachelor thesis                              *
 *             Author: Jan Findra [xfindr01 AT stud.fit.vutbr.cz]               *
 *                                    Brno 2024                                 *
 *                                                                              *
 *******************************************************************************/

#include "TMO.h"

class TMOMikamo14 : public TMO
{
public:
  /**
   * @brief Constructor
   */
  TMOMikamo14();

  /**
   * @brief Destructor
   */
  virtual ~TMOMikamo14();

  /**
   * @brief Function to compute the B-spline basis function
   * @param x input value
   * @param k degree of the spline
   * @param i index of the knot vector
   * @param t knot vector
   * @return double: value of the B-spline basis function
   */
  double B(double x, int k, int i, const std::vector<double> &t);

  /**
   * @brief Function to compute the B-spline basis function
   * @param x input value
   * @param t knot vector
   * @param c control points
   * @param k degree of the spline
   * @return double: value of the B-spline basis function
   */
  double bspline(double x, const std::vector<double> &t, const std::vector<double> &c, int k);

  /**
   * @brief Function to generate B-spline parameters
   * @param x input values
   * @param y output values
   * @param k degree of the spline
   * @return std::pair<std::vector<double>, std::vector<double>>: knot vector and control points
   */
  std::pair<std::vector<double>, std::vector<double>> generateBSplineParams(const std::vector<double> &x, const std::vector<double> &y, int k);

  /**
   * @brief Function to get the LMS to RGB matrix
   * @return cv::Mat: LMS to RGB matrix
   */
  cv::Mat getLms2RgbMat(std::vector<double> shift);

  /**
   * @brief Function to get adapted retinal illuminance, from ari or al or
   * computed from the input image
   * @return double: adapted retinal illuminance
   */
  double getAdaptedRetinalIlluminance();

  /**
   * @brief Function to get discrimination parameters for given adapted retinal
   * illuminance
   * @param I adapted retinal illuminance
   * @return vector<double>: 9 discrimination parameters
   */
  std::vector<double> getDiscriminationParams(double I);

  /**
   * @brief Function to apply two-stage model to get opponent color values
   * @param spd spectral power distribution
   * @param I adapted retinal illuminance
   * @param qn original image LMS values
   * @param zVector vector of spectral opponent color values
   * @param invM matrix to adjust the amplitudes of the cone responses
   * @param lms2rgb matrix to convert from LMS to RGB
   * @return Mat: 3 opponent color values
   */
  cv::Mat applyTwoStageModel(std::vector<double> spd, double I, std::vector<cv::Mat> qn, std::vector<cv::Mat> zVector, cv::Mat invM, cv::Mat lms2rgb);

  /**
   * @brief Function to convert RGB values to spectral power distribution
   * @param red red value
   * @param green green value
   * @param blue blue value
   * @return vector<double>: spectral power distribution
   */
  std::vector<double> RGBtoSpectrum(double red, double green, double blue);

  /**
   * @brief Function to reduce luminance based on the average luminance and
   * maximum luminance
   * @param Y luminance
   * @param YLogAvg average luminance
   * @param Ymax maximum luminance
   * @return double: reduced luminance
   */
  double luminanceReduction(double Y, double YLogAvg, double Ymax);

  /**
   * @brief Function to apply the tone mapping operator
   * @return int: 0 = success, 1 = error
   */
  virtual int Transform();

  // number of colors
  const static int colors = 7;
  enum color
  {
    White,
    Cyan,
    Magenta,
    Yellow,
    Red,
    Green,
    Blue
  };

  // visible spectrum boundries
  double lowerBound = 400.0;
  double upperBound = 700.0;

  const static int knotCount = 10; // number of knots
  const static int degree = 3;     // degree of the spline

  // 10 knot indexes on range 400nm to 700nm
  std::vector<double> indexes = {400.0, 433.3, 466.7, 500.0, 533.3, 566.7, 600.0, 633.3, 666.7, 700.0};

  // 10 knot values of 7 colors on range 400nm to 700nm for Smits 1999 RGB to Spectrum conversion generated by the matlab code from https://github.com/colour-science/smits1999
  std::vector<std::vector<double>> colorData{
      {1.0046, 1.0046, 1.0046, 1.0043, 0.9967, 0.9967, 1.0028, 1.0046, 1.0045, 1.0046},
      {0.9399, 0.9707, 1.0098, 1.0099, 1.0099, 1.0088, 0.2921, 0.0000, 0.0000, 0.0001},
      {0.9792, 1.0008, 0.8775, 0.1293, 0.0000, 0.0000, 0.7372, 1.0007, 1.0005, 1.0008},
      {0.0000, 0.0000, 0.2453, 0.7623, 1.0000, 1.0000, 1.0000, 0.9744, 0.9767, 0.9912},
      {0.1648, 0.0003, 0.0000, 0.0000, 0.0000, 0.0000, 0.6317, 1.0716, 1.0716, 1.0716},
      {0.0000, 0.0000, 0.1427, 0.8842, 1.0000, 0.9772, 0.2828, 0.0000, 0.0000, 0.0000},
      {1.0000, 1.0000, 0.7712, 0.2295, 0.0000, 0.0000, 0.0000, 0.0237, 0.0433, 0.0465}};

  // 10 knot values of 3 cone's spectral sensitivities on range 400nm to 700nm data from http://www.cvrl.org/
  std::vector<std::vector<double>> LMSsensitivities{
      {2.40836E-03, 3.20278E-02, 8.69665E-02, 2.88959E-01, 8.07299E-01, 9.97484E-01, 8.33982E-01, 3.51497E-01, 6.19706E-02, 5.89749E-03},
      {2.26991E-03, 4.74049E-02, 1.55249E-01, 4.27764E-01, 9.58589E-01, 8.54663E-01, 3.34429E-01, 4.98952E-02, 4.41197E-03, 3.65317E-04},
      {5.66498E-02, 8.70713E-01, 7.13240E-01, 1.22839E-01, 9.42821E-03, 3.86967E-04, 1.83459E-05, 0.0, 0.0, 0.0}};

  // data measured by Wandell Lab - Stanford https://cs.haifa.ac.il/hagit/courses/ist/Lectures/Tutorials/color/phosphors.mat
  std::vector<std::vector<double>> displaySpectrum = {
      {0.0, 0.025, 0.09, 0.09, 0.05, 0.045, 0.345, 3.955, 0.045, 0.205},
      {-0.0, 0.015, 0.115, 0.77, 1.42, 0.685, 0.215, 0.09, 0.0, -0.0},
      {0.175, 1.1, 1.09, 0.28, 0.075, 0.02, 0.005, 0.025, -0.005, -0.005}};

protected:
  TMODouble lm;   // luminance multiplier
  TMODouble ari;  // adapted retinal illuminance
  TMODouble al;   // average luminance
  TMODouble step; // number of bins
};
