#include "transforms.hpp"

#include <cuda_runtime.h>

#include "fractal_data.hpp"
#include "utils.hpp"

namespace fractal {

__host__ __device__ void transform_xy(float *new_x,
                                      float *new_y,
                                      unsigned short transform,
                                      coeff const *cf,
                                      float rand_d1_01,
                                      float rand_d2_01,
                                      bool rand_bool) {
  float x = cf->ac * *new_x + cf->bc * *new_y + cf->cc;
  float y = cf->dc * *new_x + cf->ec * *new_y + cf->fc;

  switch (transform) {
  case 0: /* Linear */
    *new_x = x;
    *new_y = y;
    break;
  case 1: /* Sinusoidal */
    *new_x = sinf(x);
    *new_y = sinf(y);
    break;
  case 2: /* Spherical */ {
    float r = 1.0f / (x * x + y * y);
    *new_x = r * x;
    *new_y = r * y;
    break;
  }
  case 3: /* Swirl */ {
    float r = x * x + y * y;
    *new_x = x * sinf(r) - y * cosf(r);
    *new_y = x * cosf(r) + y * sinf(r);
    break;
  }
  case 4: /* Horseshoe */ {
    float r = 1.0f / sqrtf(x * x + y * y);
    *new_x = r * (x - y) * (x + y);
    *new_y = r * 2.0f * x * y;
    break;
  }
  case 5: /* Polar */ {
    *new_x = atan2f(y, x) / pi();
    *new_y = sqrtf(x * x + y * y) - 1.0f;
    break;
  }
  case 6: /* Handkerchief */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x = r * sinf(theta + r);
    *new_y = r * cosf(theta - r);
    break;
  }
  case 7: /* Heart */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x = r * sinf(theta * r);
    *new_y = -r * cosf(theta * r);
    break;
  }
  case 8: /* Disk */ {
    float r = sqrtf(x * x + y * y) * pi();
    float theta = atan2f(y, x) / pi();
    *new_x = theta * sinf(r);
    *new_y = theta * cosf(r);
    break;
  }
  case 9: /* Spiral */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x = (1.0f / r) * (cosf(theta) + sinf(r));
    *new_y = (1.0f / r) * (sinf(theta) - cosf(r));
    break;
  }
  case 10: /* Hyperbolic */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x = sinf(theta) / r;
    *new_y = r * cosf(theta);
    break;
  }
  case 11: /* Diamond */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x = sinf(theta) * cosf(r);
    *new_y = cosf(theta) * sinf(r);
    break;
  }
  case 12: /* Ex */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    float P0 = sinf(theta + r);
    P0 = P0 * P0 * P0;
    float P1 = cosf(theta - r);
    P1 = P1 * P1 * P1;
    *new_x = r * (P0 + P1);
    *new_y = r * (P0 - P1);
    break;
  }
  case 13: /* Julia */ {
    float r = sqrtf(sqrtf(x * x + y * y));
    float theta = atan2f(y, x) * .5f;
    if (rand_bool)
      theta += pi();
    *new_x = r * cosf(theta);
    *new_y = r * sinf(theta);
    break;
  }
  case 14: /* Bent */ {
    if (x >= 0.0f && y >= 0.0f) {
      *new_x = x;
      *new_y = y;
    } else if (x < 0.0f && y >= 0.0f) {
      *new_x = 2.0f * x;
      *new_y = y;
    } else if (x >= 0.0f && y < 0.0f) {
      *new_x = x;
      *new_y = y * .5f;
    } else if (x < 0.0f && y < 0.0f) {
      *new_x = 2.0f * x;
      *new_y = y * .5f;
    }
    break;
  }
  case 15: /* Waves */ {
    *new_x = x + cf->pa1 * sinf(y / (cf->pa2 * cf->pa2));
    *new_y = y + cf->pa3 * sinf(x / (cf->pa4 * cf->pa4));
    break;
  }
  case 16: /* Fisheye */ {
    float r = 2.0f / (1.f + sqrtf(x * x + y * y));
    *new_x = r * y;
    *new_y = r * x;
    break;
  }
  case 17: /* Popcorn */ {
    *new_x = x + cf->cc * sinf(tanf(3.0f * y));
    *new_y = y + cf->fc * sinf(tanf(3.0f * x));
    break;
  }
  case 18: /* Exponential */ {
    *new_x = expf(x - 1.0f) * cosf(pi() * y);
    *new_y = expf(x - 1.0f) * sinf(pi() * y);
    break;
  }
  case 19: /* Power */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x = powf(r, sinf(theta)) * cosf(theta);
    *new_y = powf(r, sinf(theta)) * sinf(theta);
    break;
  }
  case 20: /* Cosine */ {
    *new_x = cosf(pi() * x) * coshf(y);
    *new_y = -sinf(pi() * x) * sinhf(y);
    break;
  }
  case 21: /* Rings */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    float prefix = nonnan_mod((r + cf->pa2 * cf->pa2),
                               (2.0f * cf->pa2 * cf->pa2)) - (cf->pa2 * cf->pa2) + (r * (1.0f - cf->pa2 * cf->pa2));
    *new_x = prefix * cosf(theta);
    *new_y = prefix * sinf(theta);
    break;
  }
  case 22: /* Fan */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    float t = pi() * cf->cc * cf->cc;
    if (nonnan_mod(theta, t) > (t * .5f)) {
      *new_x = r * cosf(theta - (t * .5f));
      *new_y = r * sinf(theta - (t * .5f));
    } else {
      *new_x = r * cosf(theta + (t * .5f));
      *new_y = r * sinf(theta + (t * .5f));
    }
    break;
  }
  case 23: /* Eyefish */ {
    float r = 2.0f / (1.f + sqrtf(x * x + y * y));
    *new_x = r * x;
    *new_y = r * y;
    break;
  }
  case 24: /* Bubble */ {
    float r = 4 + x * x + y * y;
    *new_x = (4.0f * x) / r;
    *new_y = (4.0f * y) / r;
    break;
  }
  case 25: /* Cylinder */ {
    *new_x = sinf(x);
    *new_y = y;
    break;
  }
  case 26: /* Tangent */ {
    *new_x = sinf(x) / cosf(y);
    *new_y = tanf(y);
    break;
  }
  case 27: /* Cross */ {
    float r = sqrtf(1.0f / ((x * x - y * y) * (x * x - y * y)));
    *new_x = x * r;
    *new_y = y * r;
    break;
  }
  case 28: /* Collatz */ {
    *new_x = .25f * (1.0f + 4.0f * x - (1.0f + 2.0f * x) * cosf(pi() * x));
    *new_y = .25f * (1.0f + 4.0f * y - (1.0f + 2.0f * y) * cosf(pi() * y));
    break;
  }
  case 29: /* Mobius */ {
    float t = (cf->pa3 * x + cf->pa4) * (cf->pa3 * x + cf->pa4) + cf->pa3 * y * cf->pa3 * y;
    *new_x = ((cf->pa1 * x + cf->pa2) * (cf->pa3 * x + cf->pa4) + cf->pa1 * cf->pa3 * y * y) / t;
    *new_y = (cf->pa1 * y * (cf->pa3 * x + cf->pa4) - cf->pa3 * y * (cf->pa1 * x + cf->pa2)) / t;
    break;
  }
  case 30: /* Blob */ {
    float r = sqrtf(x * x + y * y);
    float theta = atan2f(y, x);
    *new_x =
        r * (cf->pa2 + 0.5f * (cf->pa1 - cf->pa2) * (sinf(cf->pa3 * theta) + 1)) * cosf(theta);
    *new_y =
        r * (cf->pa2 + 0.5f * (cf->pa1 - cf->pa2) * (sinf(cf->pa3 * theta) + 1)) * sinf(theta);
    break;
  }
  case 31: /* Noise */ {
//    float theta = rand_d1_01;
//    float r = rand_d2_01;
    *new_x = rand_d1_01 * x * cosf(pi_two() * rand_d2_01);
    *new_y = rand_d1_01 * y * sinf(pi_two() * rand_d2_01);
    break;
  }
  case 32: /* Blur */ {
//    float theta = rand_d1_01;
//    float r = rand_d2_01;
    *new_x = rand_d1_01 * cosf(pi_two() * rand_d2_01);
    *new_y = rand_d1_01 * sinf(pi_two() * rand_d2_01);
    break;
  }
  case 33: /* Square */ {
    *new_x = rand_d1_01 - 0.5f;
    *new_y = rand_d2_01 - 0.5f;
    break;
  }
  case 34: /* Not Broken Waves */ {
    *new_x = x + cf->bc * sinf(y / powf(cf->cc, 2.0f));
    *new_y = y + cf->ec * sinf(x / powf(cf->fc, 2.0f));
    break;
  }
  case 35: /* something something */ {
    *new_x = y;
    *new_y = sinf(x);
    break;
  }
  default:break;
  }
}

}
