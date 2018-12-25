#include "params.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace fractal {

params::params(int argc, char **argv) : params(params::parse(argc, argv)) {}

params::params(cxxopts::ParseResult const &p) : _supersample(p["u"].as<unsigned short>()),
                                                _res_y(p["h"].as<unsigned int>() * _supersample),
                                                _res_x(p["w"].as<unsigned int>() * _supersample),
                                                _view(p["view"].as<rect>()),

                                                _rgb(p["color"].as<pix>()),
                                                _fractal_equations_count(p["n"].as<unsigned short>()),

                                                _samples(p["S"].as<unsigned int>()),
                                                _iterations(p["i"].as<unsigned short>()),
                                                _invert(p["invert"].as<bool>()),
                                                _symmetry(p["s"].as<unsigned short>()),
                                                _seed(p["r"].as<unsigned long int>()),
                                                _num_threads(p["T"].as<unsigned short>()),
                                                _gamma(p["g"].as<float>()),
                                                _file(p["o"].as<std::string>()),
                                                _transforms(p["v"].as<std::vector<unsigned short>>()),
                                                _color_file(p["color-file"].as<std::string>()),
                                                _coeff_file(p["coeff-file"].as<std::string>()),
                                                _renderer(p["renderer"].as<renderer_type>()),
                                                _rand_color(p["randomize-color"].as<bool>()),
                                                _dump_coeffs(p["d"].as<bool>()) {}

std::string params::output_file() const { return _file; }
std::string params::color_file() const { return _color_file; }
std::string params::coeff_file() const { return _coeff_file; }
std::vector<unsigned short> params::transformations() const { return _transforms; }

cxxopts::ParseResult params::parse(int argc, char **argv) {
  cxxopts::Options options(argv[0], " - fractal image generator\n");
  options.show_positional_help();
  options.custom_help("-o [FILENAME] -v [T1] -v [T2] -u [SUPERSAMPLING] [OTHER OPTIONS...]");
  options.add_options()
      ("help", "Show this information")
      ("v,variations", "Transformation list", cxxopts::value<std::vector<unsigned short>>()->default_value("29"), "<int>")
      ("o,output-file", "Output image file", cxxopts::value<std::string>()->default_value("fractal.tiff"), "<filename>")
      ("r,seed", "Random generator seed, >= 0", cxxopts::value<unsigned long int>()->default_value("42"), "<seed>")
      ("renderer", "Renderer backend", cxxopts::value<renderer_type>()->default_value("threads"), "<cuda|threads>")
      ("n,fractal-number", "Number of fractals", cxxopts::value<unsigned short>()->default_value("16"), "<int>")
      ("S,samples", "Number of passes, >= 0", cxxopts::value<unsigned int>()->default_value("10000"), "<passes>")
      ("i,iterations", "Number of iterations, >= 0", cxxopts::value<unsigned short>()->default_value("1000"), "<iters>")
      ("w,width", "Output image width, >= 0", cxxopts::value<unsigned int>()->default_value("1920"), "<width>")
      ("h,height", "Output image height, >= 0", cxxopts::value<unsigned int>()->default_value("1080"), "<height>")
      ("T,threads", "Thread number, >= 0", cxxopts::value<unsigned short>()->default_value("4"), "<threads_num>")
      ("invert", "Invert final colors")
      ("color-file", "File with color coefficients", cxxopts::value<std::string>()->default_value(""), "<filename>")
      ("coeff-file", "File with equation coefficients", cxxopts::value<std::string>()->default_value(""), "<filename>")
      ("s,symmetry", "Rotational symmetry axis, >= 1", cxxopts::value<unsigned short>()->default_value("1"), "<int>")
      ("view", "Region in model-space", cxxopts::value<rect>()->default_value("-1.7777,-1.0,3.4,2"), "<x,y,w,h>")
      ("color", "Fractal's color", cxxopts::value<pix>()->default_value("64,64,64"), "<r,g,b>")
      ("randomize-color", "Add random value to base color", cxxopts::value<bool>()->default_value("true"), "<bool>")
      ("u,supersampling", "Super sample while rendering, >= 0", cxxopts::value<unsigned short>()->default_value("2"), "<sup>")
      ("g,gamma", "Gamma factor", cxxopts::value<float>()->default_value("2.2"), "<gamma>")
      ("d,dump", "Print image coeffs", cxxopts::value<bool>()->default_value("false"), "<bool>");

  try {
    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help")) {
      print_help(options);
      exit(EXIT_SUCCESS);
    }
    return result;
  } catch (cxxopts::OptionException &ex) {
    std::cout << ex.what() << "\n";
    print_help(options);
    exit(EXIT_FAILURE);
  }
}

void params::print_help(cxxopts::Options const &options) {
  std::cout << options.help({"", "Group"}) << std::endl;
  std::cout << "Transformation list:\n"
               "0  - Linear,       1  - Sinusoidal, 2  - Spherical\n"
               "3  - Swirl,        4  - Horseshoe,  5  - Polar\n"
               "6  - Handkerchief, 7  - Heart,      8  - Disk\n"
               "9  - Spiral,       10 - Hyperbolic, 11 - Diamond\n"
               "12 - Ex,           13 - Julia,      14 - Bent\n"
               "15 - Waves,        16 - Fisheye,    17 - Popcorn\n"
               "18 - Exponential,  19 - Power,      20 - Cosine\n"
               "21 - Rings,        22 - Fan,        23 - Eyefish\n"
               "24 - Bubble,       25 - Cylinder,   26 - Tangent\n"
               "27 - Cross,        28 - Collatz,    29 - Mobius\n"
               "30 - Blob,         31 - Noise,      32 - Blur\n"
               "33 - Square,       34 - Waves2,     35 - something something\n";
  exit(EXIT_SUCCESS);
}

void parse_value(const std::string &text, rect &value) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(text);
  while (std::getline(tokenStream, token, ',')) {
    tokens.push_back(token);
  }
  float x, y, w, h;
  cxxopts::values::parse_value(tokens[0], x);
  cxxopts::values::parse_value(tokens[1], y);
  cxxopts::values::parse_value(tokens[2], w);
  cxxopts::values::parse_value(tokens[3], h);

  value = {x, y, w, h};
}

void parse_value(const std::string &text, pix &value) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(text);
  while (std::getline(tokenStream, token, ',')) {
    tokens.push_back(token);
  }
  unsigned char r, g, b;
  cxxopts::values::parse_value(tokens[0], r);
  cxxopts::values::parse_value(tokens[1], g);
  cxxopts::values::parse_value(tokens[2], b);
  value = {r, g, b, 0};
}

void parse_value(const std::string &text, renderer_type &value) {
  if (text == "threads") {
    value = renderer_type::THREADS;
  } else if (text == "cuda") {
    value = renderer_type::CUDA;
  }
}

}
