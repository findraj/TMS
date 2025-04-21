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

#include "TMOMikamo14.h"

TMOMikamo14::TMOMikamo14()
{
  SetName(L"Mikamo14");
  SetDescription(L"A tone reproduction operator for all luminance ranges considering human color perception. Two optional parameters, if both set, just ari is used.");

  lm.SetName(L"lm");
  lm.SetDescription(L"Luminance multiplier (lm); <0.0, 1000.0> (mandatory when using adapted luminance)");
  lm.SetDefault(0.0);
  lm = 0.0;
  lm.SetRange(0.0, 1000.0);
  this->Register(lm);

  ari.SetName(L"ari");
  ari.SetDescription(L"Adapted retinal illuminance (ari) in Trolands; <0.0, 1000.0> (optional)");
  ari.SetDefault(10.0);
  ari = 10.0;
  ari.SetRange(0.0, 1000.0);
  this->Register(ari);

  al.SetName(L"al");
  al.SetDescription(L"Adapted luminance (al) in cd/m^2; <0.0, 1000.0> (optional)");
  al.SetDefault(0.0);
  al = 0.0;
  al.SetRange(0.0, 1000.0);
  this->Register(al);

  step.SetName(L"step");
  step.SetDescription(L"Step size for integration; <3.0, 30.0>. Lower step size means more accurate result, but slower computation.");
  step.SetDefault(3.0);
  step = 3.0;
  step.SetRange(3.0, 30.0);
  this->Register(step);
}

TMOMikamo14::~TMOMikamo14() {}

double TMOMikamo14::B(double x, int k, int i, const std::vector<double> &t)
{
  // if k == 0, check if x is within the interval [t[i], t[i+1])
  if (k == 0)
  {
    return (t[i] <= x && x < t[i + 1]) ? 1.0 : 0.0;
  }

  double c1 = 0.0;
  // compute the first term of the B-spline basis function if the denominator is non-zero
  if (t[i + k] != t[i])
  {
    c1 = (x - t[i]) / (t[i + k] - t[i]) * B(x, k - 1, i, t);
  }

  double c2 = 0.0;
  // compute the second term of the B-spline basis function if the denominator is non-zero
  if (t[i + k + 1] != t[i + 1])
  {
    c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B(x, k - 1, i + 1, t);
  }

  return c1 + c2;
}

double TMOMikamo14::bspline(double x, const std::vector<double> &t, const std::vector<double> &c, int k)
{
  // calculate the number of basis functions
  int n = t.size() - k - 1;

  // validate the knot vector and control points size
  if (n < k + 1 || c.size() < static_cast<size_t>(n))
  {
    std::cerr << "ERROR: Invalid knot vector or control points size." << std::endl;
    exit(1);
  }

  double result = 0.0;

  // compute the weighted sum of basis functions
  for (int i = 0; i < n; ++i)
  {
    result += c[i] * B(x, k, i, t);
  }

  return result;
}

std::pair<std::vector<double>, std::vector<double>> TMOMikamo14::generateBSplineParams(const std::vector<double> &x, const std::vector<double> &y, int k)
{
  // ensure x and y vectors are of the same size and not empty
  if (x.size() != y.size() || x.empty())
  {
    std::cerr << "ERROR: x and y vectors must have the same size and cannot be empty." << std::endl;
    exit(1);
  }

  int n = x.size();  // number of data points
  int m = n + k + 1; // total number of knots

  std::vector<double> t(m); // knot vector
  // set the first k+1 knots to the first x value
  for (int i = 0; i <= k; ++i)
  {
    t[i] = x.front();
  }
  // set the internal knots based on the x values
  for (int i = k + 1; i < m - k - 1; ++i)
  {
    t[i] = x[i - (k - 1)];
  }
  // set the last k+1 knots to the last x value
  for (int i = m - k - 1; i < m; ++i)
  {
    t[i] = x.back();
  }

  std::vector<double> c = y; // control points are the y values

  return {t, c};
}

cv::Mat TMOMikamo14::getLms2RgbMat()
{
  cv::Mat lms2rgb(3, 3, CV_64F);

  // go through L-, M- and S-cone sensitivities and R, G, B display spectrum output
  for (int x = 0; x < 3; x++) // LMS
  {
    for (int y = 0; y < 3; y++) // RGB
    {
      std::pair<std::vector<double>, std::vector<double>> params1 = generateBSplineParams(indexes, LMSsensitivities[x], 3);
      std::pair<std::vector<double>, std::vector<double>> params2 = generateBSplineParams(indexes, displaySpectrum[y], 3);

      // perform integration using the B-spline basis functions
      double gamma = lowerBound;
      double sum = 0.0;

      while (gamma < upperBound)
      {
        double lms = bspline(gamma, params1.first, params1.second, 3);
        double rgb = bspline(gamma, params2.first, params2.second, 3);
        sum += lms * rgb * step;
        gamma += step;
      }
      lms2rgb.at<double>(x, y) = sum;
    }
  }

  // inverse the matrix to get RGB to LMS matrix
  return lms2rgb.inv();
}

double TMOMikamo14::getAdaptedRetinalIlluminance()
{
  // if adapted retinal illuminance is set, return it
  if (ari != 0.0 && (al == 0.0 && lm == 0.0))
  {
    return ari;
  }

  // if adapted luminance is set, return it multiplied by the pupil area
  if (al != 0.0)
  {
    double pupilDiameter = 5.697 - 0.658 * std::log10(al) + 0.07 * std::pow(std::log10(al), 2); // pupil diameter depending on the adapted luminance, equation by Blackie and Howland (1999)
    double area = M_PI * std::pow(pupilDiameter / 2, 2);                                        // area of the pupil
    return al * area;
  }

  // compute average luminance from the input image
  double luminanceSum = 0.0;
  for (int y = 0; y < pSrc->GetHeight(); y++)
  {
    for (int x = 0; x < pSrc->GetWidth(); x++)
    {
      double L = pSrc->GetLuminance(x, y);
      luminanceSum += L;
    }
  }
  double averageLuminance = luminanceSum / (pSrc->GetHeight() * pSrc->GetWidth());

  if (lm == 0.0)
  {
    std::cerr << "ERROR: Luminance multiplier is not set." << std::endl;
    exit(1);
  }

  averageLuminance *= lm;

  double diameter = 5.697 - 0.658 * std::log10(averageLuminance) + 0.07 * std::pow(std::log10(averageLuminance), 2); // pupil diameter depending on the average luminance, equation by Blackie and Howland (1999)
  double area = M_PI * std::pow(diameter / 2, 2);                                                                    // area of the pupil
  return averageLuminance * area;
}

std::vector<double> TMOMikamo14::getDiscriminationParams(double I)
{
  std::vector<double> params(9);

  params[0] = -18.3 / (1 + 7.2 * std::pow(I, -0.7)) - 0.9;   // λl(I)
  params[1] = -44.6 / (1 + 35.4 * std::pow(I, -1.2)) + 22.0; // λm(I)
  params[2] = 43.0 / (1 + 9.0 * std::pow(I, -1.5)) + 28.0;   // λs(I)

  params[3] = 6.69 / (1 + 2500 * std::pow(I, -2.65)) + 0.80; // k1(I)
  params[4] = -6.24 / (1 + 2500 * std::pow(I, -2.5)) - 0.77; // k2(I)
  params[5] = 0.36 / (1 + 50.02 * std::pow(I, -1.5)) + 0.04; // k3(I)

  params[6] = 0.24 / (1 + 50.04 * std::pow(I, -1.7)) + 0.03; // k4(I)
  params[7] = 0.42 / (1 + 1.76 * std::pow(I, -0.02)) + 0.14; // k5(I)
  params[8] = 0.15 / (1 + 2.80 * std::pow(I, -0.46)) - 0.27; // k6(I)

  return params;
}

cv::Mat TMOMikamo14::applyTwoStageModel(std::vector<double> spd, double I, std::vector<cv::Mat> qn, std::vector<cv::Mat> zVector, cv::Mat invM, cv::Mat lms2rgb)
{
  double V = 0.0;
  double Org = 0.0;
  double Oyb = 0.0;
  double L = 0.0;
  double M = 0.0;
  double S = 0.0;

  // compute the parameters for the B-spline basis functions
  std::pair<std::vector<double>, std::vector<double>> spdBSplineParams = generateBSplineParams(indexes, spd, degree);

  int i = 0; // index for the zVector
  double gamma = lowerBound;
  // go through the spectral power distribution and compute the integrated opponent color values
  while (gamma < upperBound)
  {
    // get spectral opponent color values
    cv::Mat z = zVector.at(i);
    double spd = bspline(gamma, spdBSplineParams.first, spdBSplineParams.second, degree);
    // add the opponent color values to the integrated values
    V += spd * z.at<double>(0, 0) * step;
    Org += spd * z.at<double>(1, 0) * step;
    Oyb += spd * z.at<double>(2, 0) * step;
    L += spd * qn.at(i).at<double>(0, 0) * step;
    M += spd * qn.at(i).at<double>(1, 0) * step;
    S += spd * qn.at(i).at<double>(2, 0) * step;

    i++;
    gamma += step;
  }
  // create matrix with original LMS values
  cv::Mat Qn = (cv::Mat_<double>(3, 1) << L, M, S);
  // create matrix with opponent color values
  cv::Mat opponentColor = (cv::Mat_<double>(3, 1) << V, Org, Oyb);
  // convert from opponent color space to LMS color space
  opponentColor = invM * opponentColor;
  // add LMS shift values to the LMS original values
  Qn = Qn + opponentColor;
  // convert from LMS color space to RGB color space
  Qn = lms2rgb * Qn;

  return Qn;
}

std::vector<double> TMOMikamo14::RGBtoSpectrum(double red, double green, double blue)
{
  std::vector<double> spectrum;

  for (int i = 0; i < knotCount; i++)
  {
    // get spectral power distribution for each color from the colorData
    double white_spd = colorData[0][i];
    double cyan_spd = colorData[1][i];
    double magenta_spd = colorData[2][i];
    double yellow_spd = colorData[3][i];
    double red_spd = colorData[4][i];
    double green_spd = colorData[5][i];
    double blue_spd = colorData[6][i];

    double spd = 0.0;
    // algorithm to convert RGB to spectral power distribution
    if (red <= green && red <= blue)
    {
      spd += white_spd * red;
      if (green <= blue)
      {
        spd += cyan_spd * (green - red);
        spd += blue_spd * (blue - green);
      }
      else
      {
        spd += cyan_spd * (blue - red);
        spd += green_spd * (green - blue);
      }
    }
    else if (green <= red && green <= blue)
    {
      spd += white_spd * green;
      if (red <= blue)
      {
        spd += magenta_spd * (red - green);
        spd += blue_spd * (blue - red);
      }
      else
      {
        spd += magenta_spd * (blue - green);
        spd += red_spd * (red - blue);
      }
    }
    else
    {
      spd += white_spd * blue;
      if (red <= green)
      {
        spd += yellow_spd * (red - blue);
        spd += green_spd * (green - red);
      }
      else
      {
        spd += yellow_spd * (green - blue);
        spd += red_spd * (red - green);
      }
    }
    // add the spectral power distribution to the spectrum
    spectrum.push_back(spd);
  }
  return spectrum;
}

double TMOMikamo14::luminanceReduction(double Y, double YLogAvg, double Ymax)
{
  // get key value for luminance reduction
  double alpha = 1.03 - 2.0 / (2.0 + std::log10(YLogAvg + 1.0));
  // compute reduced luminance
  double Yr = (alpha * Y) / YLogAvg;
  // compute final, normalized luminance
  double Yn = Yr / (1 + Yr);
  return Yn;
}

int TMOMikamo14::Transform()
{
  double I = getAdaptedRetinalIlluminance();
  std::vector<double> params = getDiscriminationParams(I);
  cv::Mat lms2rgb = getLms2RgbMat();
  // matrix which adjusts the amplitudes of the cone responses
  cv::Mat M = (cv::Mat_<double>(3, 3) << 0.6, 0.4, 0.0, params[3], params[4], params[5], params[6], params[7], params[8]);
  // matrix which adjust the gap in the viewing conditions and also converts from opponent color space back to LMS
  std::vector<double> adaptParams = getDiscriminationParams(150.0); // presuming viewers are in photopic conditions (150 Td)
  cv::Mat invM = (cv::Mat_<double>(3, 3) << 0.6, 0.4, 0.0, adaptParams[3], adaptParams[4], adaptParams[5], adaptParams[6], adaptParams[7], adaptParams[8]);
  invM = invM.inv();

  // compute vector of matrixes of horizontally moved spectral sensitivities

  // compute parameters for B-spline basis functions
  std::pair<std::vector<double>, std::vector<double>> LSensBSplineParams = generateBSplineParams(indexes, LMSsensitivities[0], degree);
  std::pair<std::vector<double>, std::vector<double>> MSensBSplineParams = generateBSplineParams(indexes, LMSsensitivities[1], degree);
  std::pair<std::vector<double>, std::vector<double>> SSensBSplineParams = generateBSplineParams(indexes, LMSsensitivities[2], degree);

  std::vector<cv::Mat> qn;
  std::vector<cv::Mat> zVector;

  double gamma = lowerBound;

  while (gamma < upperBound)
  {
    // matrix with original spectral sensitivities
    cv::Mat qnTmp = (cv::Mat_<double>(3, 1) << bspline(gamma, LSensBSplineParams.first, LSensBSplineParams.second, degree),
                     bspline(gamma, MSensBSplineParams.first, MSensBSplineParams.second, degree),
                     bspline(gamma, SSensBSplineParams.first, SSensBSplineParams.second, degree));
    qn.push_back(qnTmp);

    // matrix with horizontally moved spectral sensitivities
    cv::Mat CmClCs = (cv::Mat_<double>(3, 1) << bspline(gamma - params[0], LSensBSplineParams.first, LSensBSplineParams.second, degree),
                      bspline(gamma - params[1], MSensBSplineParams.first, MSensBSplineParams.second, degree),
                      bspline(gamma - params[2], SSensBSplineParams.first, SSensBSplineParams.second, degree));
    cv::Mat z = M * CmClCs;
    zVector.push_back(z);

    gamma += step;
  }

  // go through the image and apply the two-stage model
  for (int y = 0; y < pSrc->GetHeight(); y++)
  {
    for (int x = 0; x < pSrc->GetWidth(); x++)
    {
      double *srcPixel = pSrc->GetPixel(x, y);
      std::vector<double> spd = RGBtoSpectrum(srcPixel[0], srcPixel[1], srcPixel[2]);
      cv::Mat RGB = applyTwoStageModel(spd, I, qn, zVector, invM, lms2rgb);

      double *dstPixel = pDst->GetPixel(x, y);
      dstPixel[0] = std::max(RGB.at<double>(0, 0), 0.0);
      dstPixel[1] = std::max(RGB.at<double>(1, 0), 0.0);
      dstPixel[2] = std::max(RGB.at<double>(2, 0), 0.0);
    }
  }

  // luminance reduction
  double epsilon = 1e-6;
  double sumLogY = 0.0;
  int pixelCount = pDst->GetHeight() * pDst->GetWidth();
  double Ymax = 0.0;
  double rangeReduction;
  if (I > 10.0)
    rangeReduction = 1.0;
  else if (I > 1.0)
    rangeReduction = 0.5;
  else
    rangeReduction = 0.25;

  // compute sum of logarithms of luminance and maximum luminance
  for (int y = 0; y < pDst->GetHeight(); y++)
  {
    for (int x = 0; x < pDst->GetWidth(); x++)
    {
      double *pixel = pDst->GetPixel(x, y);
      double Y = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722;
      sumLogY += std::log(Y + epsilon);
      if (Y > Ymax)
      {
        Ymax = Y;
      }
    }
  }

  // compute average luminance
  double YLogAvg = std::exp(sumLogY / pixelCount);

  cv::Mat adaptedLuminance = cv::Mat::zeros(pDst->GetWidth(), pDst->GetHeight(), CV_64F);

  // go through the image and apply luminance reduction
  for (int y = 0; y < pDst->GetHeight(); y++)
  {
    for (int x = 0; x < pDst->GetWidth(); x++)
    {
      double *pixel = pDst->GetPixel(x, y);
      double Y = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722;
      double Yr = luminanceReduction(Y, YLogAvg, Ymax);
      adaptedLuminance.at<double>(x, y) = Yr;
    }
  }

  pDst->Convert(TMO_Yxy);

  for (int y = 0; y < pDst->GetHeight(); y++)
  {
    for (int x = 0; x < pDst->GetWidth(); x++)
    {
      pDst->GetPixel(x, y)[0] = adaptedLuminance.at<double>(x, y) * rangeReduction;
    }
  }

  return 0;
}
