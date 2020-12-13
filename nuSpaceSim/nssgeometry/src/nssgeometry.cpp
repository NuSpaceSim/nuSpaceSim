#include <iostream>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <vector>
// #include "H5Cpp.h"

#define TO_STREAM(stream, variable) \
  (stream) << #variable "\n" //": " << (variable) << std::endl

// using namespace H5;

using namespace std;

namespace py = pybind11;

float heaviside(double x) {
  if (x < 0)
    return 0;
  else
    return 1.0;
}

class ev_data_t {
public:
  float ts, ps, rs, ds, ttsv, cttsv, ptsv, ttsn, cttsn, btsn, lpl, tnsv, ctnsv;
  int   ke;
};

class Event {
public:
  float thetaS, phiS, raS, decS, thetaTrSubV, costhetaTrSubV, phiTrSubV,
      thetaTrSubN, costhetaTrSubN, betaTrSubN, losPathLen, thetaNSubV,
      costhetaNSubV;

  Event()
      : thetaS(0)
      , phiS(0)
      , raS(0)
      , decS(0)
      , thetaTrSubV(0)
      , costhetaTrSubV(1.)
      , phiTrSubV(0)
      , thetaTrSubN(0)
      , costhetaTrSubN(1.)
      , betaTrSubN(0)
      , losPathLen(0)
      , thetaNSubV(0)
      , costhetaNSubV(1.){};
  Event(float ranthetaS,
        float ranphiS,
        float ranraS,
        float randecS,
        float ranthetaTrSubV,
        float cosranthetaTrSubV,
        float ranphiTrSubV,
        float ranthetaTrSubN,
        float cosranthetaTrSubN,
        float ranbetaTrSubN,
        float ranlosPathLen,
        float ranthetaNSubV,
        float cosranthetaNSubV)
      : thetaS(ranthetaS)
      , phiS(ranphiS)
      , raS(ranraS)
      , decS(randecS)
      , thetaTrSubV(ranthetaTrSubV)
      , costhetaTrSubV(cosranthetaTrSubV)
      , phiTrSubV(ranphiTrSubV)
      , thetaTrSubN(ranthetaTrSubN)
      , costhetaTrSubN(cosranthetaTrSubN)
      , betaTrSubN(ranbetaTrSubN)
      , losPathLen(ranlosPathLen)
      , thetaNSubV(ranthetaNSubV)
      , costhetaNSubV(cosranthetaNSubV){};
};

class Geom_params {
private:
  // static bool init;
  std::random_device               rd;
  std::mt19937                     gen; // gen(rd());
  std::uniform_real_distribution<> dis; // dis(0.0, 1.0);

  float GeoPi;
  float earthRadius;
  float earthRadiusSqrd;
  float detAlt;
  float detRA;
  float detDec;
  float alphaHorizon;
  float alphaMax;
  float alphaMin;
  float minChordLen;
  float maxChordLen;
  float maxThetaS;
  float minThetaS;
  float cosOfMaxThetaS;
  float cosOfMinThetaS;
  float maxPhiS;
  float minPhiS;
  float minLOSpathLen;
  float minLOSpathLenSqrd;
  float minLOSpathLenCubed;
  float maxLOSpathLen;
  float maxLOSpathLenSqrd;
  float maxLOSpathLenCubed;
  float radEplusDetAlt;
  float radEplusDetAltSqrd;
  float maxThetaTrSubV; // This is the maximum theta_Ch for optical Cherenkov
                        // detection
  float sinOfMaxThetaTrSubV;
  float normThetaTrSubV;
  float normPhiTrSubV;
  float normPhiS;
  float normThetaS;
  float pdfnorm;

public:
  float              mcnorm;
  float              geom_factor;
  Event              localevent;
  bool               keepLocalEv;
  py::array_t<Event> evArray;
  py::array_t<bool>  evMasknpArray; // For a numpy array of events

  Geom_params(float, float, float, float, float, float, float, float);
  Geom_params(float,
              float,
              float,
              float,
              float,
              float,
              float,
              float,
              float,
              float);
  ~Geom_params() {}
  void set_det_coord(float, float);
  void gen_traj();
  void gen_traj_from_set(float, float, float, float);
  void get_local_event();
  void run_geo_dmc_from_num_traj_hdf5(int);
  // void run_geo_dmc_from_ran_array_hdf5(py::array_t<float>);
  void run_geo_dmc_from_num_traj_nparray(int);
  void run_geo_dmc_from_ran_array_nparray(py::array_t<float>);
  // void make_event_mask_array();
  void  print_event_from_array(int);
  void  print_geom_factor();
  float get_det_ra() { return detRA; }
  float get_det_dec() { return detDec; }
};

// This constructor is for limb viewing

Geom_params::Geom_params(float radE,
                         float detalt,
                         float detra,
                         float detdec,
                         float delAlpha,
                         float maxsepangle,
                         float delAziAng,
                         float ParamPi)
    : gen(rd())
    , dis(0.0,
          nextafter(
              1.0,
              std::numeric_limits<float>::max())) // Closed interval [0.0,1.0]
{
  float bracketForNormThetaS;

  GeoPi           = ParamPi;
  earthRadius     = radE;
  earthRadiusSqrd = earthRadius * earthRadius;

  detAlt             = detalt;
  radEplusDetAlt     = earthRadius + detalt;
  radEplusDetAltSqrd = radEplusDetAlt * radEplusDetAlt;

  detRA  = detra * (GeoPi / 180.0); // Convert to radians
  detDec = detdec * (GeoPi / 180.0);

  alphaHorizon = 0.5 * GeoPi - acos(earthRadius / radEplusDetAlt);
  alphaMin = alphaHorizon - delAlpha; // delAlpha should already be in radians
  minChordLen = 2. * sqrt(earthRadiusSqrd -
                          radEplusDetAltSqrd * sin(alphaMin) * sin(alphaMin));

  minLOSpathLen      = radEplusDetAlt * cos(alphaMin) - 0.5 * minChordLen;
  minLOSpathLenSqrd  = minLOSpathLen * minLOSpathLen;
  minLOSpathLenCubed = minLOSpathLen * minLOSpathLen * minLOSpathLen;

  maxThetaS = acos(earthRadius / radEplusDetAlt);
  minThetaS = acos((radEplusDetAltSqrd + earthRadiusSqrd - minLOSpathLenSqrd) *
                   1. / (2. * radEplusDetAlt * earthRadius));
  cosOfMaxThetaS = cos(maxThetaS);
  cosOfMinThetaS = cos(minThetaS);

  maxLOSpathLen      = sqrt(radEplusDetAltSqrd - earthRadiusSqrd);
  maxLOSpathLenSqrd  = maxLOSpathLen * maxLOSpathLen;
  maxLOSpathLenCubed = maxLOSpathLen * maxLOSpathLen * maxLOSpathLen;

  maxThetaTrSubV      = maxsepangle;
  sinOfMaxThetaTrSubV = sin(maxThetaTrSubV);

  maxPhiS = 0.5 * delAziAng;
  minPhiS = -0.5 * delAziAng;

  normThetaTrSubV = 2. / (sinOfMaxThetaTrSubV * sinOfMaxThetaTrSubV);
  normPhiTrSubV   = 1. / (2. * GeoPi);
  normPhiS        = 1. / (maxPhiS - minPhiS);

  bracketForNormThetaS =
      (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen -
      (1. / 3.) * maxLOSpathLenCubed -
      (radEplusDetAltSqrd - earthRadiusSqrd) * minLOSpathLen +
      (1. / 3.) * minLOSpathLenCubed;
  normThetaS = 2. * radEplusDetAlt * earthRadiusSqrd / bracketForNormThetaS;

  pdfnorm = normThetaTrSubV * normPhiTrSubV * normPhiS * normThetaS;
  mcnorm  = earthRadiusSqrd /
           pdfnorm; // Still need to divide by the number of trajectories

  geom_factor = 0.0;
}

// This constructor is for off-limb viewing

Geom_params::Geom_params(float radE,
                         float detalt,
                         float detra,
                         float detdec,
                         float nadmin,
                         float nadmax,
                         float maxsepangle,
                         float minAziAng,
                         float maxAziAng,
                         float ParamPi)
    : gen(rd())
    , dis(0.0,
          nextafter(
              1.0,
              std::numeric_limits<float>::max())) // Closed interval [0.0,1.0]
{
  float bracketForNormThetaS;

  GeoPi           = ParamPi;
  earthRadius     = radE;
  earthRadiusSqrd = earthRadius * earthRadius;

  detAlt             = detalt;
  radEplusDetAlt     = earthRadius + detalt;
  radEplusDetAltSqrd = radEplusDetAlt * radEplusDetAlt;

  detRA  = detra * (GeoPi / 180.0); // Convert to radians
  detDec = detdec * (GeoPi / 180.0);

  alphaMin    = nadmin; // Should already be in radians
  alphaMax    = nadmax;
  minChordLen = 2. * sqrt(earthRadiusSqrd -
                          radEplusDetAltSqrd * sin(alphaMin) * sin(alphaMin));
  maxChordLen = 2. * sqrt(earthRadiusSqrd -
                          radEplusDetAltSqrd * sin(alphaMax) * sin(alphaMax));

  minLOSpathLen      = radEplusDetAlt * cos(alphaMin) - 0.5 * minChordLen;
  minLOSpathLenSqrd  = minLOSpathLen * minLOSpathLen;
  minLOSpathLenCubed = minLOSpathLen * minLOSpathLen * minLOSpathLen;

  maxLOSpathLen      = radEplusDetAlt * cos(alphaMax) - 0.5 * maxChordLen;
  maxLOSpathLenSqrd  = maxLOSpathLen * maxLOSpathLen;
  maxLOSpathLenCubed = maxLOSpathLen * maxLOSpathLen * maxLOSpathLen;

  minThetaS = acos((radEplusDetAltSqrd + earthRadiusSqrd - minLOSpathLenSqrd) *
                   1. / (2. * radEplusDetAlt * earthRadius));
  maxThetaS = acos((radEplusDetAltSqrd + earthRadiusSqrd - maxLOSpathLenSqrd) *
                   1. / (2. * radEplusDetAlt * earthRadius));
  cosOfMaxThetaS = cos(maxThetaS);
  cosOfMinThetaS = cos(minThetaS);

  maxThetaTrSubV      = maxsepangle;
  sinOfMaxThetaTrSubV = sin(maxThetaTrSubV);

  maxPhiS = maxAziAng;
  minPhiS = minAziAng;

  normThetaTrSubV = 2. / (sinOfMaxThetaTrSubV * sinOfMaxThetaTrSubV);
  normPhiTrSubV   = 1. / (2. * GeoPi);
  normPhiS        = 1. / (maxPhiS - minPhiS);

  bracketForNormThetaS =
      (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen -
      (1. / 3.) * maxLOSpathLenCubed -
      (radEplusDetAltSqrd - earthRadiusSqrd) * minLOSpathLen +
      (1. / 3.) * minLOSpathLenCubed;
  normThetaS = 2. * radEplusDetAlt * earthRadiusSqrd / bracketForNormThetaS;

  pdfnorm = normThetaTrSubV * normPhiTrSubV * normPhiS * normThetaS;
  mcnorm  = earthRadiusSqrd /
           pdfnorm; // Still need to divide by the number of trajectories

  geom_factor = 0.0;
}

/*Geom_params::~Geom_params(){
  // Need to close the file and deallocate memory for the array?
  }*/

void Geom_params::set_det_coord(float detra, float detdec) {
  detRA  = detra * (GeoPi / 180.0); // Convert to radians
  detDec = detdec * (GeoPi / 180.0);
}

void Geom_params::gen_traj() {
  bool geoKeep = true;

  float rthetaS, rphiS, rraS, rdecS, rthetaTrSubV, rcosthetaTrSubV, rphiTrSubV,
      rthetaTrSubN, rbetaTrSubN, rlosPathLen;
  float u1, u2, u3, u4;
  float b, q, r, s, t, psi, discriminant;
  float v1, v2, v3;
  float maxLOSpathLenCubed = maxLOSpathLen * maxLOSpathLen * maxLOSpathLen;
  float minLOSpathLenCubed = minLOSpathLen * minLOSpathLen * minLOSpathLen;
  float rcosthetaS, rvsqrd;
  float rcosthetaNSubV, rthetaNSubV;
  float rcosthetaTrSubN;
  float rsindecS;
  float rxS, ryS;

  u1 = dis(gen);

  rthetaTrSubV    = asin(sinOfMaxThetaTrSubV * sqrt(u1));
  rcosthetaTrSubV = cos(rthetaTrSubV);

  u2 = dis(gen);

  rphiTrSubV = 2. * GeoPi * u2;

  u3 = dis(gen);

  rphiS = (maxPhiS - minPhiS) * u3 + minPhiS;

  // Generate theta_s (the colatitude on the surface of the Earth in the
  // detector nadir perspective)

  b = 3. * (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen -
      maxLOSpathLenCubed -
      3. * (radEplusDetAltSqrd - earthRadiusSqrd) * minLOSpathLen +
      minLOSpathLenCubed;

  q = -1. * (radEplusDetAltSqrd - earthRadiusSqrd);

  u4 = dis(gen);

  r = -1.5 * (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen +
      0.5 * maxLOSpathLenCubed + 0.5 * b * u4;

  discriminant = q * q * q + r * r;

  if (discriminant <= 0) {
    psi = acos(r / sqrt(-1. * q * q * q));
    v1  = 2. * sqrt(-1. * q) * cos(psi / 3.);
    v2  = 2. * sqrt(-1. * q) * cos((psi + 2. * GeoPi) / 3.);
    v3  = 2. * sqrt(-1. * q) * cos((psi + 4. * GeoPi) / 3.);

    if ((v1 > 0) && (v1 >= minLOSpathLen) && (v1 <= maxLOSpathLen))
      rlosPathLen = v1;
    if ((v2 > 0) && (v2 >= minLOSpathLen) && (v2 <= maxLOSpathLen))
      rlosPathLen = v2;
    if ((v3 > 0) && (v3 >= minLOSpathLen) && (v3 <= maxLOSpathLen))
      rlosPathLen = v3;
  } else {
    s = pow((r + sqrt(discriminant)), (1. / 3));
    t = pow((r - sqrt(discriminant)), (1. / 3));

    rlosPathLen = s + t;
  }

  rvsqrd     = rlosPathLen * rlosPathLen;
  rcosthetaS = (radEplusDetAltSqrd + earthRadiusSqrd - rvsqrd) /
               (2. * earthRadius * radEplusDetAlt);
  rthetaS = acos(rcosthetaS);

  rcosthetaNSubV = (radEplusDetAltSqrd - earthRadiusSqrd - rvsqrd) /
                   (2. * earthRadius * rlosPathLen);

  rthetaNSubV = acos(rcosthetaNSubV);

  rcosthetaTrSubN = cos(rthetaTrSubV) * rcosthetaNSubV -
                    sin(rthetaTrSubV) * sin(rthetaNSubV) * cos(rphiTrSubV);

  rthetaTrSubN = acos(rcosthetaTrSubN);

  if (rcosthetaTrSubN < 0.0)
    geoKeep = false; // Trajectories going into the ground

  rbetaTrSubN = (0.5 * GeoPi - rthetaTrSubN) * (180.0 / GeoPi);

  rsindecS = sin(detDec) * rcosthetaS - cos(detDec) * sin(rthetaS) * cos(rphiS);
  rdecS    = asin(rsindecS) * 180.0 / GeoPi;

  rxS = sin(detDec) * cos(detRA) * sin(rthetaS) * cos(rphiS) -
        sin(detRA) * sin(rthetaS) * sin(rphiS) +
        cos(detDec) * cos(detRA) * cos(rthetaS);

  ryS = sin(detDec) * sin(detRA) * sin(rthetaS) * cos(rphiS) +
        cos(detRA) * sin(rthetaS) * sin(rphiS) +
        cos(detDec) * sin(detRA) * cos(rthetaS);

  rraS = atan2(ryS, rxS) * 180.0 / GeoPi; // Convert to Degrees

  if (rraS < 0.0) rraS += 360.0;

  this->localevent = Event(rthetaS,
                           rphiS,
                           rraS,
                           rdecS,
                           rthetaTrSubV,
                           rcosthetaTrSubV,
                           rphiTrSubV,
                           rthetaTrSubN,
                           rcosthetaTrSubN,
                           rbetaTrSubN,
                           rlosPathLen,
                           rthetaNSubV,
                           rcosthetaNSubV);

  keepLocalEv = geoKeep;
}

void Geom_params::gen_traj_from_set(float u1, float u2, float u3, float u4) {
  bool geoKeep = true;

  float rthetaS, rphiS, rraS, rdecS, rthetaTrSubV, rcosthetaTrSubV, rphiTrSubV,
      rthetaTrSubN, rbetaTrSubN, rlosPathLen;
  float b, q, r, s, t, psi, discriminant;
  float v1, v2, v3;
  float rcosthetaS, rvsqrd;
  float rcosthetaNSubV, rthetaNSubV;
  float rcosthetaTrSubN;
  // float cosdecS;
  float rsindecS;
  float rxS, ryS;

  rthetaTrSubV    = asin(sinOfMaxThetaTrSubV * sqrt(u1));
  rcosthetaTrSubV = cos(rthetaTrSubV);

  rphiTrSubV = 2. * GeoPi * u2;

  rphiS = (maxPhiS - minPhiS) * u3 + minPhiS;

  // Generate theta_s (the colatitude on the surface of the Earth in the
  // detector nadir perspective)

  b = 3. * (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen -
      maxLOSpathLenCubed -
      3. * (radEplusDetAltSqrd - earthRadiusSqrd) * minLOSpathLen +
      minLOSpathLenCubed;

  q = -1. * (radEplusDetAltSqrd - earthRadiusSqrd);

  r = -1.5 * (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen +
      0.5 * maxLOSpathLenCubed + 0.5 * b * u4;

  discriminant = q * q * q + r * r;

  if (discriminant <= 0) {
    psi = acos(r / sqrt(-1. * q * q * q));
    v1  = 2. * sqrt(-1. * q) * cos(psi / 3.);
    v2  = 2. * sqrt(-1. * q) * cos((psi + 2. * GeoPi) / 3.);
    v3  = 2. * sqrt(-1. * q) * cos((psi + 4. * GeoPi) / 3.);

    if ((v1 > 0) && (v1 >= minLOSpathLen) && (v1 <= maxLOSpathLen))
      rlosPathLen = v1;
    if ((v2 > 0) && (v2 >= minLOSpathLen) && (v2 <= maxLOSpathLen))
      rlosPathLen = v2;
    if ((v3 > 0) && (v3 >= minLOSpathLen) && (v3 <= maxLOSpathLen))
      rlosPathLen = v3;
  } else {
    s = pow((r + sqrt(discriminant)), (1. / 3));
    t = pow((r - sqrt(discriminant)), (1. / 3));

    rlosPathLen = s + t;
  }

  rvsqrd     = rlosPathLen * rlosPathLen;
  rcosthetaS = (radEplusDetAltSqrd + earthRadiusSqrd - rvsqrd) /
               (2. * earthRadius * radEplusDetAlt);
  rthetaS = acos(rcosthetaS);

  rcosthetaNSubV = (radEplusDetAltSqrd - earthRadiusSqrd - rvsqrd) /
                   (2. * earthRadius * rlosPathLen);

  rthetaNSubV = acos(rcosthetaNSubV);

  rcosthetaTrSubN = cos(rthetaTrSubV) * rcosthetaNSubV -
                    sin(rthetaTrSubV) * sin(rthetaNSubV) * cos(rphiTrSubV);

  rthetaTrSubN = acos(rcosthetaTrSubN);

  if (rcosthetaTrSubN < 0.0)
    geoKeep = false; // Trajectories going into the ground

  rbetaTrSubN = (0.5 * GeoPi - rthetaTrSubN) * (180.0 / GeoPi);

  rsindecS = sin(detDec) * rcosthetaS - cos(detDec) * sin(rthetaS) * cos(rphiS);
  rdecS    = asin(rsindecS) * 180.0 / GeoPi;

  rxS = sin(detDec) * cos(detRA) * sin(rthetaS) * cos(rphiS) -
        sin(detRA) * sin(rthetaS) * sin(rphiS) +
        cos(detDec) * cos(detRA) * cos(rthetaS);

  ryS = sin(detDec) * sin(detRA) * sin(rthetaS) * cos(rphiS) +
        cos(detRA) * sin(rthetaS) * sin(rphiS) +
        cos(detDec) * sin(detRA) * cos(rthetaS);

  rraS = atan2(ryS, rxS) * 180.0 / GeoPi; // Convert to Degrees

  if (rraS < 0.0) rraS += 360.0;

  this->localevent = Event(rthetaS,
                           rphiS,
                           rraS,
                           rdecS,
                           rthetaTrSubV,
                           rcosthetaTrSubV,
                           rphiTrSubV,
                           rthetaTrSubN,
                           rcosthetaTrSubN,
                           rbetaTrSubN,
                           rlosPathLen,
                           rthetaNSubV,
                           rcosthetaNSubV);

  keepLocalEv = geoKeep;
}

void Geom_params::get_local_event() {
  Event       e        = this->localevent;
  const char* kestring = (keepLocalEv ? "true" : "false");

  printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %s\n",
         e.thetaS,
         e.phiS,
         e.raS,
         e.decS,
         e.thetaTrSubV,
         e.costhetaTrSubV,
         e.phiTrSubV,
         e.thetaTrSubN,
         e.costhetaTrSubN,
         e.betaTrSubN,
         e.losPathLen,
         e.thetaNSubV,
         e.costhetaNSubV,
         kestring);
}

// void Geom_params::run_geo_dmc_from_num_traj_hdf5(int numEvents){
//  const H5std_string FILE_NAME("Resources/nssSimulationData.h5");
//  //const H5std_string FILE_NAME("nssSimulationData.h5");
//  const H5std_string DATASET_NAME("nssEventDSet");
//  const H5std_string ATTR_NAME("EventInfo");
//  const H5std_string MEMBER1("thetaS");
//  const H5std_string MEMBER2("phiS");
//  const H5std_string MEMBER3("raS");
//  const H5std_string MEMBER4("decS");
//  const H5std_string MEMBER5("thetaTrSubV");
//  const H5std_string MEMBER6("costhetaTrSubV");
//  const H5std_string MEMBER7("phiTrSubV");
//  const H5std_string MEMBER8("thetaTrSubN");
//  const H5std_string MEMBER9("costhetaTrSubN");
//  const H5std_string MEMBER10("betaTrSubN");
//  const H5std_string MEMBER11("losPathLen");
//  const H5std_string MEMBER12("thetaNSubV");
//  const H5std_string MEMBER13("costhetaNSubV");
//  const H5std_string MEMBER14("keepEv");
//  const int RANK = 1;
//  const int numDataMembers = 14; // For the HDF5 File implementation

//  int i;

//  const float thetaCh = GeoPi*(1.5/180.); // For the geometry factor for the
//  trajectories only monte carlo. const float costhetaCh = cos(thetaCh);

//  Event thisevent;
//  bool keepThisEvent;

//  // This block is for the HDF5 File implementation

//  stringstream namess, eventnamess;

//  string namestr, eventstr;
//  string geostr = "geom_factor = ";
//  std::vector<string> attrstr;

//  std::vector<ev_data_t> evData;

//  evData.reserve(numEvents);

//  TO_STREAM(eventnamess,thisevent);
//  getline(eventnamess,eventstr);

//  geom_factor = 0.0;

//  for (i = 0; i < numEvents; i++){
//    gen_traj();
//    thisevent = this->localevent;
//    keepThisEvent = keepLocalEv;

//    ev_data_t tempEvent;

//    // This block is for the HDF5 File implementation

//    tempEvent.ts = thisevent.thetaS;
//    tempEvent.ps = thisevent.phiS;
//    tempEvent.rs = thisevent.raS;
//    tempEvent.ds = thisevent.decS;
//    tempEvent.ttsv = thisevent.thetaTrSubV;
//    tempEvent.cttsv = thisevent.costhetaTrSubV;
//    tempEvent.ptsv = thisevent.phiTrSubV;
//    tempEvent.ttsn = thisevent.thetaTrSubN;
//    tempEvent.cttsn = thisevent.costhetaTrSubN;
//    tempEvent.btsn = thisevent.betaTrSubN;
//    tempEvent.lpl = thisevent.losPathLen;
//    tempEvent.tnsv = thisevent.thetaNSubV;
//    tempEvent.ctnsv = thisevent.costhetaNSubV;
//    tempEvent.ke = keepThisEvent;

//    evData.push_back(tempEvent);

//    if (keepThisEvent)
//      geom_factor += (heaviside(thisevent.costhetaTrSubV -
//      costhetaCh)*thisevent.costhetaTrSubN)
//	/(thisevent.costhetaNSubV*thisevent.costhetaTrSubV);
//  }

//  geom_factor *= (1./numEvents)*mcnorm;

//  // This block is for the attribute that includes the complete event
//  information in the HDF5 File implementation

//  /*for (i = 0; i < numDataMembers; i++){
//    switch(i)
//      {
//      case 0:
//	{
//	  TO_STREAM(namess,thisevent.thetaS);
//	  break;
//	}
//      case 1:
//	{
//	  TO_STREAM(namess,thisevent.phiS);
//	  break;
//	}
//      case 2:
//	{
//	  TO_STREAM(namess,thisevent.raS);
//	  break;
//	}
//      case 3:
//	{
//	  TO_STREAM(namess,thisevent.decS);
//	  break;
//	}
//      case 4:
//	{
//	  TO_STREAM(namess,thisevent.thetaTrSubV);
//	  break;
//	}
//      case 5:
//	{
//	  TO_STREAM(namess,thisevent.costhetaTrSubV);
//	  break;
//	}
//      case 6:
//	{
//	  TO_STREAM(namess,thisevent.phiTrSubV);
//	  break;
//	}
//      case 7:
//	{
//	  TO_STREAM(namess,thisevent.thetaTrSubN);
//	  break;
//	}
//      case 8:
//	{
//	  TO_STREAM(namess,thisevent.costhetaTrSubN);
//	  break;
//	}
//      case 9:
//	{
//	  TO_STREAM(namess,thisevent.betaTrSubN);
//	  break;
//	}
//      case 10:
//	{
//	  TO_STREAM(namess,thisevent.losPathLen);
//	  break;
//	}
//      case 11:
//	{
//	  TO_STREAM(namess,thisevent.thetaNSubV);
//	  break;
//	}
//      case 12:
//	{
//	  TO_STREAM(namess,thisevent.costhetaNSubV);
//	  break;
//	}
//      case 13:
//	{
//	  TO_STREAM(namess,thisevent.keepEv);
//	  break;
//	}
//      default:
//	{
//	  throw std::runtime_error("Not a valid index.\n");
//	}
//      }
//    getline(namess,namestr);
//    namestr.erase(0,eventstr.length()+1);
//    attrstr.push_back(namestr);
//    }*/

//  geostr.append(to_string(geom_factor));
//  geostr.append(" km^2 sr");
//  attrstr.push_back(geostr);

//  try{
//    Exception::dontPrint();

//    H5File *file = new H5File(FILE_NAME, H5F_ACC_TRUNC);

//    hsize_t dim[] = {static_cast<hsize_t>(numEvents)};
//    DataSpace *dataspace = new DataSpace(RANK, dim);

//    hid_t boolenumtype = H5Tcreate(H5T_ENUM, sizeof(bool));

//    short val;
//    H5Tenum_insert(boolenumtype, "FALSE", (val=0,&val));
//    H5Tenum_insert(boolenumtype, "TRUE", (val=1,&val));

//    CompType evtype(sizeof(ev_data_t));
//    evtype.insertMember(MEMBER1, HOFFSET(ev_data_t, ts),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER2, HOFFSET(ev_data_t,
//    ps), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER3,
//    HOFFSET(ev_data_t, rs), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER4, HOFFSET(ev_data_t, ds),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER5, HOFFSET(ev_data_t,
//    ttsv), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER6,
//    HOFFSET(ev_data_t, cttsv), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER7, HOFFSET(ev_data_t, ptsv),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER8, HOFFSET(ev_data_t,
//    ttsn), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER9,
//    HOFFSET(ev_data_t, cttsn), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER10, HOFFSET(ev_data_t, btsn),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER11, HOFFSET(ev_data_t,
//    lpl), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER12,
//    HOFFSET(ev_data_t, tnsv), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER13, HOFFSET(ev_data_t, ctnsv),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER14, HOFFSET(ev_data_t,
//    ke), boolenumtype);

//    DataSet *dataset = new DataSet(file->createDataSet(DATASET_NAME, evtype,
//    *dataspace));

//    delete dataset;
//    delete dataspace;
//    delete file;

//    file = new H5File(FILE_NAME, H5F_ACC_RDWR);
//    dataset = new DataSet(file->openDataSet(DATASET_NAME));
//    dataset->write(evData.data(),evtype); // For vector evData

//    hsize_t attr_dims[1]; //= { (int) dims[1] }; // Dimensions of attribute
//    dataset attr_dims[0] = attrstr.size();

//    DataSpace *attr_dataspace = new DataSpace(1, attr_dims);

//    // Create new string datatype for attribute

//    //StrType strdatatype(PredType::C_S1,20);
//    StrType strdatatype(PredType::C_S1, H5T_VARIABLE);

//    Attribute *attr = new
//    Attribute(dataset->createAttribute(ATTR_NAME,strdatatype,*attr_dataspace));

//    vector<const char *> cStrArray;
//    for (int index = 0; index < attrstr.size(); ++index)
//    cStrArray.push_back(attrstr[index].c_str());

//    attr->write(strdatatype, (void*)&cStrArray[0]);

//    delete attr;
//    delete dataset;
//    delete file;
//  }

//  // catch failure caused by the H5File operations

//  catch(FileIException error){
//    error.printErrorStack();
//    //return -1;
//    //throw -1;
//  }

//  // catch failure caused by the DataSet operations

//  catch(DataSetIException error){
//    error.printErrorStack();
//    //return -1;
//    //throw -1;
//  }

//  // catch failure caused by the DataSpace operations

//  catch(DataSpaceIException error){
//    error.printErrorStack();
//    //return -1;
//    //throw -1;
//  }

//  // catch failure caused by invalid index

//  catch(runtime_error &e){
//    cout << "Error in the execution: " << e.what() << endl;
//  }
//}


void Geom_params::run_geo_dmc_from_num_traj_nparray(int numEvents) {
  int i;

  size_t idx;

  const float thetaCh =
      GeoPi *
      (1.5 /
       180.); // For the geometry factor for the trajectories only monte carlo.
  const float costhetaCh = cos(thetaCh);

  Event thisevent;
  bool  keepThisEvent;

  // cout << "Here!\n";

  geom_factor = 0.0;

  // This block is for reading from the numpy array of uniform random floats and
  // for making the event numpy array

  auto result =
      py::array(py::buffer_info(nullptr,
                                sizeof(Event),
                                py::format_descriptor<Event>::format(),
                                1,
                                { numEvents },
                                { sizeof(Event) }));

  auto resultm = py::array(py::buffer_info(nullptr,
                                           sizeof(bool),
                                           py::format_descriptor<bool>::value,
                                           1,
                                           { numEvents },
                                           { sizeof(bool) }));

  auto buf2 = result.request();
  auto bufm = resultm.request();

  auto ptr  = static_cast<Event*>(buf2.ptr);
  auto mptr = static_cast<bool*>(bufm.ptr);

  geom_factor = 0.0;

  for (idx = 0; idx < numEvents; idx++) {
    gen_traj();

    // This block is for the numpy array implementation

    ptr[idx]  = localevent;
    mptr[idx] = keepLocalEv;

    if (keepLocalEv)
      geom_factor += (heaviside(localevent.costhetaTrSubV - costhetaCh) *
                      localevent.costhetaTrSubN) /
                     (localevent.costhetaNSubV * localevent.costhetaTrSubV);
  }
  geom_factor *= (1. / numEvents) * mcnorm;

  // This block is for the numpy array implementation

  evArray       = result;
  evMasknpArray = resultm;
}

// void Geom_params::run_geo_dmc_from_ran_array_hdf5(py::array_t<float> input){
//  const H5std_string FILE_NAME("Resources/nssSimulationData.h5");
//  //const H5std_string FILE_NAME("nssSimulationData.h5");
//  const H5std_string DATASET_NAME("nssEventDSet");
//  const H5std_string ATTR_NAME("EventInfo");
//  const H5std_string MEMBER1("thetaS");
//  const H5std_string MEMBER2("phiS");
//  const H5std_string MEMBER3("raS");
//  const H5std_string MEMBER4("decS");
//  const H5std_string MEMBER5("thetaTrSubV");
//  const H5std_string MEMBER6("costhetaTrSubV");
//  const H5std_string MEMBER7("phiTrSubV");
//  const H5std_string MEMBER8("thetaTrSubN");
//  const H5std_string MEMBER9("costhetaTrSubN");
//  const H5std_string MEMBER10("betaTrSubN");
//  const H5std_string MEMBER11("losPathLen");
//  const H5std_string MEMBER12("thetaNSubV");
//  const H5std_string MEMBER13("costhetaNSubV");
//  const H5std_string MEMBER14("keepEv");
//  const int RANK = 1;
//  const int numDataMembers = 14; // For the HDF5 File implementation

//  // This block is for reading from the numpy array of uniform random floats

//  auto buf1 = input.request();

//  if (buf1.shape[1] != 4)
//    throw std::runtime_error("Number of columns must be four.");

//  size_t idx, numEvents = buf1.shape[0];

//  const float thetaCh = GeoPi*(1.5/180.); // For the geometry factor for the
//  trajectories only monte carlo. const float costhetaCh = cos(thetaCh);

//  float u1, u2, u3, u4;

//  stringstream namess, eventnamess;

//  Event thisevent;
//  bool keepThisEvent;

//  string namestr, eventstr;
//  string geostr = "geom_factor = ";
//  vector<string> attrstr;

//  std::vector<ev_data_t> evData;

//  evData.reserve(numEvents);

//  // This block is for reading from the numpy array of uniform random floats

//  float *ptr1 = (float *) buf1.ptr;

//  geom_factor = 0.0;

//  for (idx = 0; idx < numEvents; idx++){
//    u1 = *(ptr1 + idx*buf1.shape[1]);
//    u2 = *(ptr1 + idx*buf1.shape[1] + 1);
//    u3 = *(ptr1 + idx*buf1.shape[1] + 2);
//    u4 = *(ptr1 + idx*buf1.shape[1] + 3);

//    gen_traj_from_set(u1,u2,u3,u4);
//    thisevent = this->localevent;
//    keepThisEvent = keepLocalEv;

//    ev_data_t tempEvent;

//    // This block is for the HDF5 File implementation

//    tempEvent.ts = thisevent.thetaS;
//    tempEvent.ps = thisevent.phiS;
//    tempEvent.rs = thisevent.raS;
//    tempEvent.ds = thisevent.decS;
//    tempEvent.ttsv = thisevent.thetaTrSubV;
//    tempEvent.cttsv = thisevent.costhetaTrSubV;
//    tempEvent.ptsv = thisevent.phiTrSubV;
//    tempEvent.ttsn = thisevent.thetaTrSubN;
//    tempEvent.cttsn = thisevent.costhetaTrSubN;
//    tempEvent.btsn = thisevent.betaTrSubN;
//    tempEvent.lpl = thisevent.losPathLen;
//    tempEvent.tnsv = thisevent.thetaNSubV;
//    tempEvent.ctnsv = thisevent.costhetaNSubV;
//    tempEvent.ke = keepThisEvent;

//    evData.push_back(tempEvent);

//    // This block is for the attribute that includes the complete event
//    information in the HDF5 File implementation

//    if (keepThisEvent)
//      geom_factor += (heaviside(thisevent.costhetaTrSubV -
//      costhetaCh)*thisevent.costhetaTrSubN)
//	/(thisevent.costhetaNSubV*thisevent.costhetaTrSubV);
//  }
//  geom_factor *= (1./numEvents)*mcnorm;

//  // This block is for the HDF5 File implementation

//  /*for (i = 0; i < numDataMembers; i++){
//    switch(i)
//      {
//      case 0:
//	{
//	  TO_STREAM(namess,thisevent.thetaS);
//	  break;
//	}
//      case 1:
//	{
//	  TO_STREAM(namess,thisevent.phiS);
//	  break;
//	}
//      case 2:
//	{
//	  TO_STREAM(namess,thisevent.raS);
//	  break;
//	}
//      case 3:
//	{
//	  TO_STREAM(namess,thisevent.decS);
//	  break;
//	}
//      case 4:
//	{
//	  TO_STREAM(namess,thisevent.thetaTrSubV);
//	  break;
//	}
//      case 5:
//	{
//	  TO_STREAM(namess,thisevent.costhetaTrSubV);
//	  break;
//	}
//      case 6:
//	{
//	  TO_STREAM(namess,thisevent.phiTrSubV);
//	  break;
//	}
//      case 7:
//	{
//	  TO_STREAM(namess,thisevent.thetaTrSubN);
//	  break;
//	}
//      case 8:
//	{
//	  TO_STREAM(namess,thisevent.costhetaTrSubN);
//	  break;
//	}
//      case 9:
//	{
//	  TO_STREAM(namess,thisevent.betaTrSubN);
//	  break;
//	}
//      case 10:
//	{
//	  TO_STREAM(namess,thisevent.losPathLen);
//	  break;
//	}
//      case 11:
//	{
//	  TO_STREAM(namess,thisevent.thetaNSubV);
//	  break;
//	}
//      case 12:
//	{
//	  TO_STREAM(namess,thisevent.costhetaNSubV);
//	  break;
//	}
//      case 13:
//	{
//	  TO_STREAM(namess,thisevent.keepEv);
//	  break;
//	}
//      default:
//	{
//	  throw runtime_error("Not a valid index.\n");
//	}
//      }
//    getline(namess,namestr);
//    namestr.erase(0,eventstr.length()+1);
//    attrstr.push_back(namestr);
//    }*/

//  geostr.append(to_string(geom_factor));
//  geostr.append(" km^2 sr");
//  attrstr.push_back(geostr);

//  try{
//    Exception::dontPrint();

//    H5File *file = new H5File(FILE_NAME, H5F_ACC_TRUNC);

//    hsize_t dim[] = {static_cast<hsize_t>(numEvents)};
//    DataSpace *dataspace = new DataSpace(RANK, dim);

//    hid_t boolenumtype = H5Tcreate(H5T_ENUM, sizeof(bool));

//    short val;
//    H5Tenum_insert(boolenumtype, "FALSE", (val=0,&val));
//    H5Tenum_insert(boolenumtype, "TRUE", (val=1,&val));

//    CompType evtype(sizeof(ev_data_t));
//    evtype.insertMember(MEMBER1, HOFFSET(ev_data_t, ts),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER2, HOFFSET(ev_data_t,
//    ps), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER3,
//    HOFFSET(ev_data_t, rs), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER4, HOFFSET(ev_data_t, ds),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER5, HOFFSET(ev_data_t,
//    ttsv), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER6,
//    HOFFSET(ev_data_t, cttsv), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER7, HOFFSET(ev_data_t, ptsv),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER8, HOFFSET(ev_data_t,
//    ttsn), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER9,
//    HOFFSET(ev_data_t, cttsn), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER10, HOFFSET(ev_data_t, btsn),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER11, HOFFSET(ev_data_t,
//    lpl), PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER12,
//    HOFFSET(ev_data_t, tnsv), PredType::NATIVE_FLOAT);
//    evtype.insertMember(MEMBER13, HOFFSET(ev_data_t, ctnsv),
//    PredType::NATIVE_FLOAT); evtype.insertMember(MEMBER14, HOFFSET(ev_data_t,
//    ke), boolenumtype);

//    DataSet *dataset = new DataSet(file->createDataSet(DATASET_NAME, evtype,
//    *dataspace));

//    delete dataset;
//    delete dataspace;
//    delete file;

//    file = new H5File(FILE_NAME, H5F_ACC_RDWR);
//    dataset = new DataSet(file->openDataSet(DATASET_NAME));
//    dataset->write(evData.data(),evtype);

//    hsize_t attr_dims[1]; //= { (int) dims[1] }; // Dimensions of attribute
//    dataset attr_dims[0] = attrstr.size();

//    DataSpace *attr_dataspace = new DataSpace(1, attr_dims);

//    // Create new string datatype for attribute

//    //StrType strdatatype(PredType::C_S1,20);
//    StrType strdatatype(PredType::C_S1, H5T_VARIABLE);

//    Attribute *attr = new
//    Attribute(dataset->createAttribute(ATTR_NAME,strdatatype,*attr_dataspace));

//    vector<const char *> cStrArray;
//    for (int index = 0; index < attrstr.size(); ++index)
//    cStrArray.push_back(attrstr[index].c_str());

//    attr->write(strdatatype, (void*)&cStrArray[0]);

//    delete attr;
//    delete dataset;
//    delete file;
//  }

//  // catch failure caused by the H5File operations

//  catch(FileIException error){
//    error.printErrorStack();
//    //return -1;
//    //throw -1;
//  }

//  // catch failure caused by the DataSet operations

//  catch(DataSetIException error){
//    error.printErrorStack();
//    //return -1;
//    //throw -1;
//  }

//  // catch failure caused by the DataSpace operations

//  catch(DataSpaceIException error){
//    error.printErrorStack();
//    //return -1;
//    //throw -1;
//  }

//  // catch failure caused by invalid index

//  catch(runtime_error &e){
//    cout << "Error in the execution: " << e.what() << endl;
//  }
//}

void Geom_params::run_geo_dmc_from_ran_array_nparray(py::array_t<float> input) {
  // This block is for reading from the numpy array of uniform random floats

  auto buf1 = input.request();

  if (buf1.shape[1] != 4)
    throw std::runtime_error("Number of columns must be four.");

  size_t idx, numEvents = buf1.shape[0];

  const float thetaCh =
      GeoPi *
      (1.5 /
       180.); // For the geometry factor for the trajectories only monte carlo.
  const float costhetaCh = cos(thetaCh);

  float u1, u2, u3, u4;

  Event thisevent;
  bool  keepThisEvent;

  // This block is for reading from the numpy array of uniform random floats and
  // for making the event numpy array

  auto result =
      py::array(py::buffer_info(nullptr,
                                sizeof(Event),
                                py::format_descriptor<Event>::format(),
                                1,
                                { buf1.shape[0] },
                                { sizeof(Event) }));

  auto resultm = py::array(py::buffer_info(nullptr,
                                           sizeof(bool),
                                           py::format_descriptor<bool>::value,
                                           1,
                                           { buf1.shape[0] },
                                           { sizeof(bool) }));

  auto buf2 = result.request();
  auto bufm = resultm.request();

  auto ptr  = static_cast<Event*>(buf2.ptr);
  auto mptr = static_cast<bool*>(bufm.ptr);

  float* ptr1 = (float*)buf1.ptr;

  geom_factor = 0.0;

  for (idx = 0; idx < numEvents; idx++) {
    u1 = *(ptr1 + idx * buf1.shape[1]);
    u2 = *(ptr1 + idx * buf1.shape[1] + 1);
    u3 = *(ptr1 + idx * buf1.shape[1] + 2);
    u4 = *(ptr1 + idx * buf1.shape[1] + 3);

    gen_traj_from_set(u1, u2, u3, u4);

    // This block is for the numpy array implementation

    ptr[idx]  = localevent;
    mptr[idx] = keepLocalEv;

    if (keepLocalEv)
      geom_factor += (heaviside(localevent.costhetaTrSubV - costhetaCh) *
                      localevent.costhetaTrSubN) /
                     (localevent.costhetaNSubV * localevent.costhetaTrSubV);
  }
  geom_factor *= (1. / numEvents) * mcnorm;

  // This block is for the numpy array implementation

  evArray       = result;
  evMasknpArray = resultm;
}

/*void Geom_params::make_event_mask_array(){

  auto result = py::array(py::buffer_info(nullptr, sizeof(bool),
                      py::format_descriptor<bool>::value,
                      1,{evMask.size},{sizeof(bool)}));

  auto buf = result.request();

  auto ptr = static_cast<bool*>(buf.ptr);

  for (size_t idx = 0; idx < evMask.size; idx++)
    ptr[idx] = evMask[idx];

  evMasknpArray = result;
}*/

void Geom_params::print_event_from_array(int numElement) {
  auto  buf = evArray.request();
  auto  ptr = static_cast<Event*>(buf.ptr);
  Event e   = ptr[numElement];

  auto        buf1     = evMasknpArray.request();
  auto        ptr1     = static_cast<bool*>(buf1.ptr);
  bool        keepE    = ptr1[numElement];
  const char* kestring = (keepE ? "true" : "false");

  printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %s\n",
         e.thetaS,
         e.phiS,
         e.raS,
         e.decS,
         e.thetaTrSubV,
         e.costhetaTrSubV,
         e.phiTrSubV,
         e.thetaTrSubN,
         e.costhetaTrSubN,
         e.betaTrSubN,
         e.losPathLen,
         e.thetaNSubV,
         e.costhetaNSubV,
         kestring);
}

void Geom_params::print_geom_factor() {
  printf("Geometry Factor: %f km^2 sr\n", this->geom_factor);
}

PYBIND11_MODULE(nssgeometry, m) {

  m.doc() = "pybind11 nssgeometry plugin";

  m.def("heaviside", &heaviside, "Heaviside Function");

  py::class_<Event>(m, "Event")
      .def(py::init<>())
      .def(py::init<float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float>(),
           py::arg("ranthetaS"),
           py::arg("ranphiS"),
           py::arg("ranraS"),
           py::arg("randecS"),
           py::arg("ranthetaTrSubV"),
           py::arg("cosranthetaTrSubV"),
           py::arg("ranphiTrSubV"),
           py::arg("ranthetaTrSubN"),
           py::arg("cosranthetaTrSubN"),
           py::arg("ranbetaTrSubN"),
           py::arg("ranlosPathLen"),
           py::arg("ranthetaNSubV"),
           py::arg("cosranthetaNSubV"))
      .def_readwrite("thetaS", &Event::thetaS)
      .def_readwrite("phiS", &Event::phiS)
      .def_readwrite("raS", &Event::raS)
      .def_readwrite("decS", &Event::decS)
      .def_readwrite("thetaTrSubV", &Event::thetaTrSubV)
      .def_readwrite("costhetaTrSubV", &Event::costhetaTrSubV)
      .def_readwrite("phiTrSubV", &Event::phiTrSubV)
      .def_readwrite("thetaTrSubN", &Event::thetaTrSubN)
      .def_readwrite("costhetaTrSubN", &Event::costhetaTrSubN)
      .def_readwrite("betaTrSubN", &Event::betaTrSubN)
      .def_readwrite("losPathLen", &Event::losPathLen)
      .def_readwrite("thetaNSubV", &Event::thetaNSubV)
      .def_readwrite("costhetaNSubV", &Event::costhetaNSubV);

  // This block is for the numpy array implementation

  PYBIND11_NUMPY_DTYPE(Event,
                       thetaS,
                       phiS,
                       raS,
                       decS,
                       thetaTrSubV,
                       costhetaTrSubV,
                       phiTrSubV,
                       thetaTrSubN,
                       costhetaTrSubN,
                       betaTrSubN,
                       losPathLen,
                       thetaNSubV,
                       costhetaNSubV);

  py::class_<Geom_params>(m, "Geom_params")
      .def(py::init<float, float, float, float, float, float, float, float>(),
           py::arg("radE")        = 6371.0,
           py::arg("detalt")      = 525.0,
           py::arg("detra")       = 0.0,
           py::arg("detdec")      = 0.0,
           py::arg("delAlpha")    = 4. * atan(1.) * (7.0 / 180.0),
           py::arg("maxsepangle") = 4. * atan(1.) * (3.0 / 180.),
           py::arg("delAziAng")   = 2.0 * 4. * atan(1.),
           py::arg("ParamPi")     = 4. * atan(1.))
      .def(py::init<float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float>(),
           py::arg("radE")        = 6371.0,
           py::arg("detalt")      = 525.0,
           py::arg("detra")       = 0.0,
           py::arg("detdec")      = 0.0,
           py::arg("nadmin")      = 0.0,
           py::arg("nadmax")      = 2. * atan(1.),
           py::arg("maxsepangle") = 4. * atan(1.) * (3.0 / 180.),
           py::arg("minAziAng")   = 0.0,
           py::arg("maxAziAng")   = 2. * 4. * atan(1.),
           py::arg("ParamPi")     = 4. * atan(1.))
      .def("set_det_coord", &Geom_params::set_det_coord)
      .def("gen_traj", &Geom_params::gen_traj)
      .def("gen_traj_from_set", &Geom_params::gen_traj_from_set)
      .def("get_local_event", &Geom_params::get_local_event)
      // .def("run_geo_dmc_from_num_traj_hdf5",&Geom_params::run_geo_dmc_from_num_traj_hdf5)
      .def("run_geo_dmc_from_num_traj_nparray",
           &Geom_params::run_geo_dmc_from_num_traj_nparray)
      // .def("run_geo_dmc_from_ran_array_hdf5",&Geom_params::run_geo_dmc_from_ran_array_hdf5)
      .def("run_geo_dmc_from_ran_array_nparray",
           &Geom_params::run_geo_dmc_from_ran_array_nparray)
      .def("print_event_from_array", &Geom_params::print_event_from_array)
      .def("print_geom_factor", &Geom_params::print_geom_factor)
      .def("get_det_ra", &Geom_params::get_det_ra)
      .def("get_det_dec", &Geom_params::get_det_dec)
      //.def("make_event_mask_array",&Geom_params::make_event_mask_array)
      .def_readwrite("mcnorm", &Geom_params::mcnorm)
      .def_readwrite("geom_factor", &Geom_params::geom_factor)
      .def_readwrite("localevent", &Geom_params::localevent)
      .def_readwrite("evArray", &Geom_params::evArray)
      .def_readwrite("evMasknpArray",
                     &Geom_params::evMasknpArray); // This block is for the
                                                   // numpy array implementation
}
