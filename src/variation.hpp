#ifndef VARIATION_HPP
#define VARIATION_HPP

#include <vector_types.h>

enum class variation_type : short { LINEAR = 0 };

struct float6 {
  float3 x;
  float3 y;
};

class Variation {
public:
  //  Variation(variation_type vt, float6 affineCoeff, float6 postCoeff,
  //            float4 parametricCoeff);

private:
};

#endif // VARIATION_HPP
