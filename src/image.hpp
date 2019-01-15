#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <gsl/gsl>
#include <memory>
#include <vector_types.h>

struct pixel {
  short3 rgb;
  float density;
};

class Image {
public:
  Image(unsigned int width, unsigned int height);
  explicit Image(gsl::multi_span<pixel> buf);

  pixel *data() const { return _pixels.get(); }

private:
  std::unique_ptr<pixel> _pixels;
};

#endif // IMAGE_HPP
