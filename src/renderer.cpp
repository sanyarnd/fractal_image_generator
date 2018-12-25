#include "renderer.hpp"

#include <cmath>
#include <thread>
#include <iostream>

#include "fractal_data.hpp"
#include "image.hpp"
#include "utils.hpp"
#include "transforms.hpp"

namespace fractal {
void render_thread_proc(image &image,
                        flame_coeffs const &fc,
                        std::vector<std::mutex> &thread_mutex,
                        unsigned long int seed,
                        unsigned int samples,
                        unsigned short iter_per_sample,
                        std::vector<unsigned short> const &transformations,
                        rect view,
                        unsigned short symmetry);

image render_threads(unsigned int width,
                     unsigned int height,
                     unsigned short num_threads,
                     flame_coeffs const &fc,
                     unsigned long int seed,
                     unsigned int samples,
                     unsigned short iter_per_sample,
                     std::vector<unsigned short> const &transformations,
                     rect const &view,
                     unsigned short symmetry) {
  using namespace std::chrono;

  std::cout << "CPU rendering, running " << num_threads << " threads\n\n";
  auto start = system_clock::now();
  std::vector<std::thread> threads(num_threads);
  std::vector<std::mutex> thread_mutex(width * height);

  image buf(width, height);
  for (size_t i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(&render_thread_proc,
                             std::ref(buf),
                             std::ref(fc),
                             std::ref(thread_mutex),
                             seed * (i + 1), // different seed for each thread
                             samples / num_threads,
                             iter_per_sample,
                             std::ref(transformations),
                             view,
                             symmetry);
  }
  std::cout << "All threads spawned\n";

  for (unsigned short i = 0; i < num_threads; i++) {
    std::cout << "Joining thread #" << (i + 1) << std::endl;
    threads[i].join();
  }
  auto end = system_clock::now();
  auto elapsed = duration_cast<milliseconds>(end - start);
  std::cout << "rendering completed!\n";
  std::cout << "render time: " << elapsed.count() << "ms\n\n";

  return buf;
}

void render_thread_proc(image &image,
                        flame_coeffs const &fc,
                        std::vector<std::mutex> &thread_mutex,
                        unsigned long int seed,
                        unsigned int samples,
                        unsigned short iter_per_sample,
                        std::vector<unsigned short> const &transformations,
                        rect view,
                        unsigned short symmetry) {
  rand_gen_std rg(seed);

  for (unsigned short num = 0; num < samples; ++num) {
    float new_x = rg.rand_float(view.x, view.x + view.w);
    float new_y = rg.rand_float(view.y, view.y + view.h);

    for (unsigned short step = 0; step < iter_per_sample; ++step) {
      unsigned int iter_num = num * iter_per_sample + step;

      auto ci = static_cast<unsigned short>(rg.rand_uint(0, std::numeric_limits<int>::max()) % fc.size());

      // on each step we take the next transformation from the list
      const unsigned short transform = transformations[iter_num % transformations.size()];
      // get new new_x/new_y
      transform_xy(&new_x, &new_y, transform, fc.data() + ci, rg.rand_float(), rg.rand_float(), rg.random_bool());

      float theta2 = 0.0;
      for (size_t s = 0; s < symmetry; theta2 += pi_two() / symmetry, ++s) {
        float x_rot = new_x * cosf(theta2) - new_y * sinf(theta2);
        float y_rot = new_x * sinf(theta2) + new_y * cosf(theta2);

        if (view.contains(x_rot, y_rot)) {
          auto x1 = static_cast<unsigned int>(map_range(view.x, view.x_end(), 0, image.width(), x_rot));
          auto y1 = static_cast<unsigned int>(map_range(view.y, view.y_end(), 0, image.height(), y_rot));

          if (image.contains(x1, y1)) {
            // lock pixel mutex
            std::lock_guard<std::mutex> locker(*mat_offset_elem(thread_mutex, y1, x1, image.width()));
            pix *p = image.pixel(y1, x1);

            unsigned char r = fc[ci].r;
            unsigned char g = fc[ci].g;
            unsigned char b = fc[ci].b;
            // add base color on the first iteration
            if (p->inc_cnt == 0) {
              *p = {r, g, b, p->inc_cnt + 1};
            } else {
              *p = {static_cast<unsigned char>((p->r + r) / 2),
                    static_cast<unsigned char>((p->g + g) / 2),
                    static_cast<unsigned char>((p->b + b) / 2), p->inc_cnt + 1};
            }
          }
        }
      }
    }
  }
}
}

