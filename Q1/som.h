#ifndef CS4P80_SELF_ORGANIZING_MAP_H
#define CS4P80_SELF_ORGANIZING_MAP_H

/*
	COSC 4P80 - Assignment 2
	Self Organizing Map Structure
	@author Tennyson Demchuk (td16qg@brocku.ca) | St#: 6190532
	@date 03.22.2021
*/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

/*======================*/
/*** TYPE DEFINITIONS ***/
/*======================*/

// type aliases
typedef uint64_t			ulong;
typedef uint32_t			uint;
typedef unsigned char		uchar;
typedef std::string			str;
typedef std::vector<float>	fvec;
typedef std::vector<uint>	uivec;

// define custome namespace
namespace som {
	// float array wrapper
	template <int N>
	struct vec {
		vec(const fvec& args) {
			assert(args.size() == N);
			for (int i = 0; i < N; i++) e[i] = args[i];
		}
		float e[N];
	};
	template <int N> using grid = std::vector<vec<N>>;

	/*======================*/
	/*** HELPER FUNCTIONS ***/
	/*======================*/

	// computes random float value between 0 and 1
	inline float randFloat() {
		return (float)rand() / (float)RAND_MAX;
	}

	// sets grid size, vector dimension, and zeroes out all vectors
	template <int N> void initGrid(int size, grid<N>& g) {
		g.clear();
		g.reserve(size);
		fvec blank;
		for (int i = 0; i < N; i++) blank.push_back(0);
		for (int i = 0; i < size; i++) {
			g.push_back(blank);
		}
	}

	// randomly initializes grid array with vectors
	template<int N> void randomizeGrid(grid<N>& g) {
		for (auto& e : g) {
			for (int i = 0; i < N; i++) {
				e.e[i] = randFloat();
			}
		}
	}

	/*====================*/
	/*** SOM DEFINITION ***/
	/*====================*/
	template<int N> class SOM {
	private:
		typedef struct neighbour {		// neighbouring vector wrapper
			neighbour(uint i, float d) { index = i; dist = d; }
			uint index;
			float dist;
		} neighbour;
		typedef std::vector<neighbour> nvec;

		grid<N>&	input;		// input vector
		grid<N>&	lattice;	// 2D lattice of nodes
		uivec		indexlst;	// list of indices into input array
		uint		ind;		// current inputlst index
		const uint	ldim;		// dimension of square lattice
		const uint	max_epoch;	// maximum # training iterations
		uint		epoch;		// training iteration
		const float nrad_0;		// initial neighbourhood radius
		float		nrad;		// neighbourhood radius
		const float nlambda;	// neigbourhood update lambda constant
		float		lr;			// learning rate
		const float lr_0;		// initial learning rate
		nvec		nbr;		// vectors in neighbourhood
		inline uint index(uint x, uint y) { return y * ldim + x; }	// computes the 1d index from two points
		inline float dist(int x1, int y1, int x2, int y2) {			// returns the cartesian distance between two nodes in the lattice
			return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
		}
		inline float dist(const vec<N>& a, const vec<N>& b) {			// returns the euclidean distance between two color vectors
			float result = 0;
			for (int i = 0; i < N; i++) result += std::pow(b.e[i] - a.e[i], 2);
			return std::sqrt(result);
		}
	public:
		// constructor
		SOM() = delete;
		SOM(grid<N>& in, grid<N>& l, int latticeDim, float learningRate = 0.1f, int maxEpochs = 100, float radius = -1) :
			input(in),
			lattice(l),
			ldim(latticeDim),
			nrad_0((radius == -1 ? (ldim / 2.0f) : radius)),
			lr_0(learningRate),
			max_epoch(maxEpochs),
			nlambda((float)max_epoch / std::log2(nrad_0))
		{
			assert(latticeDim > 0, "Dimension of square lattice must be greater than 0.");
			assert(lr_0 >= 0, "Learning rate must be non-negative");
			assert(max_epoch > 0, "Must have more than one training iteration");
			assert(nrad_0 > 0, "Initial neighbourhood radius must be greater than 0");
			assert(nrad_0 <= (ldim / 2.0f), "Initial neighbourhood radius must be no greater than lattice radius");
			epoch = 1;
			nrad = 0.0f;
			lr = 0.0f;
			nbr.reserve(lattice.size());	// neighbourhood will be no larger than the lattice itself
			ind = 0;
			for (uint i = 0; i < input.size(); i++) indexlst.push_back(i);
		}

		// execute a single epoch of training
		bool train() {
			if (epoch >= max_epoch) return true;
			// 1. randomly choose vector from training data - http://www.ai-junkie.com/ann/som/som1.html
			if (ind == 0) std::random_shuffle(indexlst.begin(), indexlst.end());
			uint in = indexlst[ind];
			ind = (ind + 1) % indexlst.size();
			// 2. every node is examined to calculate who's weights are most like the input vector - the winner is the Best Matching Unit (BMU)
			uint bmu_x = 0, bmu_y = 0;
			float cdist = dist(lattice[0], input[in]);
			for (uint y = 0; y < ldim; y++) {
				for (uint x = 1; x < ldim; x++) {
					float tmp = dist(lattice[index(x, y)], input[in]);
					if (tmp < cdist) {
						cdist = tmp;
						bmu_x = x;
						bmu_y = y;
					}
				}
			}
			// 3. The radius of the neighbourhood of the BMU is calculated. This value starts large, typically initialized as the 'radius' (dim)
			//		of the lattice, and diminishes each time step. All nodes within this radius are deemed to be within the BMU's neighbourhood
			nrad = nrad_0 * std::exp(-(float)epoch / nlambda);
			nbr.clear();
			float ndist;
			for (uint y = 0; y < ldim; y++) {
				for (uint x = 0; x < ldim; x++) {
					if ((ndist = dist(x, y, bmu_x, bmu_y)) <= nrad) {
						nbr.emplace_back(index(x, y), ndist);			// store indices of all nodes in neighbourhood
					}
				}
			}
			// 4. Each neighbouring node's (the nodes found in step 3) weights are updated to make them more like the input vector. The close the
			//		node is to the BMU, the more it's weights get altered
			lr = lr_0 * std::exp(-(float)epoch / max_epoch);
			for (int n = 0; n < nbr.size(); n++) {
				float lrtheta = lr * (float)std::exp(-1 * ((nbr[n].dist * nbr[n].dist) / (2.0f * nrad * nrad)));
				for (int i = 0; i < N; i++) {
					lattice[nbr[n].index].e[i] += lrtheta * (input[in].e[i] - lattice[nbr[n].index].e[i]);
				}
			}
			++epoch;
			return false;
		}
	};
}
#endif