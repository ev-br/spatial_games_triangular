#include "evolve.h"
#include<cmath>


void fake_evolve(std::vector<int>& field, double b, int num_steps)
{
    for (std::size_t j=0; j < field.size(); ++j){
        field[j] = field[j] == 0 ? 1 : 0;
    }

}


void evolve_field(std::vector<int>& field, double b, int num_steps)
{
    int size = static_cast<int>(sqrt(field.size()));

	std::vector<int> scores(size*size, -1);
	std::vector<int> prevField(size*size);
	std::copy(field.begin(), field.end(), prevField.begin());

	//Scores
	for (int k = 0; k < size; k++) {
        int y = k / size; // Row
        int x = k % size; // Col

		for (int i = -1; i <= 1; i++) //Row
		{
			for (int j = -1; j <= 1; j++) //Col
			{
				int memberIndex = (x + i + size) % size + size * (y + j + size) % size;

				scores[k] += (1 - field[memberIndex]);
			}
		}

		if (field[k] == 1)
		{
			scores[k] *= b;
		}
    }
	

	//Strategy
	for (int k = 0; k < size; k++) {
		int y = k / size; // Row
		int x = k % size; // Col

		int bestStrategyIndex = k;

		for (int i = -1; i <= 1; i++) //Row
		{
			for (int j = -1; j <= 1; j++) //Col
			{
				int memberIndex = (x + i + size) % size +
							  size * (y + j + size) % size;

				if (scores[bestStrategyIndex] < scores[memberIndex]) 
				{
					bestStrategyIndex = memberIndex;
				}
			}
		}

		field[k] = prevField[bestStrategyIndex];
	}

	scores.clear();
	prevField.clear();
}
