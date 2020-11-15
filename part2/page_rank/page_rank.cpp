#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

	// initialize vertex weights to uniform probability. Double
	// precision scores are used to avoid underflow for large graphs

	int numNodes = num_nodes(g);
	double equal_prob = 1.0 / numNodes;
	double *score_new = (double *)malloc(sizeof(double) * numNodes);
	double *diff = (double *)malloc(sizeof(double) * numNodes);
	#pragma omp parallel for
	for (int i = 0; i < numNodes; ++i) {
		solution[i] = equal_prob;
	}

	while (!convergence) {
		double global_diff = 0.0;
		#pragma omp parallel for reduction(+:global_diff)
		for (int i=0; i<num_nodes(g); i++) {
		// Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
			const Vertex* start = incoming_begin(g, i);
			const Vertex* end = incoming_end(g, i);
			int j_outDegree = outgoing_size(g, i);
			if (!j_outDegree) {
				for (Vertex* j = start; j != end; j++) {
					score_new[i] += solution[j] / j_outDegree;
					score_new[i] = (damping * score_new[i]) + (1.0-damping)/ numNodes;
				}
			} 
			else {
				score_new[i] += damping * solution[i] / numNodes;
			}
			diff[i] = abs(score_new[i] - solution[i]);
			global_diff += diff[i];
			solution[i] = score_new[i];
			score_new[i] = 0;
		}
		if (global_diff < convergence) {
			free(score_new);
			free(diff);
			break;
		}
	}
	/*
	For PP students: Implement the page rank algorithm here.  You
	are expected to parallelize the algorithm using openMP.  Your
	solution may need to allocate (and free) temporary arrays.

	Basic page rank pseudocode is provided below to get you started:
	// initialization: see example code above
	score_old[vi] = 1/numNodes;

	while (!converged) {

	// compute score_new[vi] for all nodes vi:
	score_new[vi] = sum over all nodes vj reachable from incoming edges
						{ score_old[vj] / number of edges leaving vj  }
	score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

	score_new[vi] += sum over all nodes v in graph with no outgoing edges
						{ damping * score_old[v] / numNodes }

	// compute how much per-node scores have changed
	// quit once algorithm has converged

	global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
	converged = (global_diff < convergence)
	}
	*/
  
}
