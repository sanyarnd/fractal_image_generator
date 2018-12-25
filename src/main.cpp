#include <iostream>
#include <memory>

#include "image.hpp"
#include "params.hpp"
#include "renderer.hpp"
#include "renderer_cuda.hpp"

fractal::image render(fractal::params const &args, fractal::flame_coeffs const &fc) {
  switch (args.image_renderer()) {
  case fractal::renderer_type::THREADS: {
    return fractal::render_threads(args.width(),
                                   args.height(),
                                   args.num_threads(),
                                   fc,
                                   args.seed(),
                                   args.samples(),
                                   args.iter_per_sample(),
                                   args.transformations(),
                                   args.view(),
                                   args.symmetry());
  }
  case fractal::renderer_type::CUDA: {
    return fractal::render_cuda(args.width(),
                                args.height(),
                                fc,
                                args.seed(),
                                args.samples(),
                                args.iter_per_sample(),
                                args.transformations(),
                                args.view(),
                                args.symmetry());
  }
  default: {
    std::cout << "Rendering method is not yet implemented!\n";
    exit(EXIT_FAILURE);
  }
  }
}

int main(int argc, char **argv) {
  fractal::params args(argc, argv);

  // create coefficients
  fractal::flame_coeffs fc = args.coeff_file().empty() ?
                             fractal::flame_coeffs(args.fractal_equations_count(),
                                                   args.seed(),
                                                   args.color(),
                                                   args.randomize_color()) :
                             fractal::flame_coeffs(args.fractal_equations_count(),
                                                   args.color_file(),
                                                   args.coeff_file());
  if (args.dump_coefficients()) {
    fc.print_coeffs();
  }

  // render fractal
  fractal::image f = render(args, fc);
  // gamma correction
  f.gamma_log(args.gamma());
  // save to file
  f.save(args.output_file(), args.supersampling(), args.invert_color());

  return 0;
}

