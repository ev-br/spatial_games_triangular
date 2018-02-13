#include "evolve.h"
#include<cmath>

void evolve_field(std::vector<int>& field, int num_steps)
{
    int L = static_cast<int>(sqrt(field.size()));
    field[L/2] = field[L/2] == 1 ? 0 : 1;
}
