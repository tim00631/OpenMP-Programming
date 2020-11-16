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
	// #pragma omp parallel for
	// for (int i = 0; i < numNodes; ++i) {
	// 	solution[i] = equal_prob;
	// }
	#pragma omp parallel for
	for (int i = 0; i < numNodes; ++i) {
		score_new[i] = equal_prob;
	}
	bool converged = false;
	while (!converged) {
		#pragma omp parallel for
		for (int i = 0; i < numNodes; ++i) {
			solution[i] = score_new[i];
			score_new[i] = 0;
		}
		double sum = 0.0;	
		#pragma omp parallel for reduction(+:sum)	
		for(int v=0; v<numNodes; v++) {
			if (!outgoing_size(g, v)) {
				sum += damping * solution[v] / numNodes;
			}
		}
		#pragma omp parallel for
		for (int i=0;i<numNodes;i++){
			// const Vertex* start = incoming_begin(g, i);
			// const Vertex* end = incoming_end(g, i);
			int start = g->incoming_starts[i];
			int end = start + incoming_size(g,i);
			// 	score_new[vi] = sum over all nodes vj reachable from incoming edges
			//  					{ score_old[vj] / number of edges leaving vj  }	
			double inherit = 0.0;
			// #pragma omp for reduction(+:inherit)
			for (int j=start; j!=end; j++) {
				Vertex node_j = g->incoming_edges[j]; // get node_j from edge array
				int j_outdegree = outgoing_size(g, node_j); //get j's out degrees
				if(j_outdegree) {
					inherit += solution[node_j] / j_outdegree;
				}
			}
			score_new[i] += inherit;
		}
		#pragma omp parallel for
		for (int i=0;i<numNodes;i++){
			// score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
			score_new[i] = (damping * score_new[i]) + (1.0-damping)/ numNodes;
			// score_new[vi] += sum over all nodes v in graph with no outgoing edges
			// 						{ damping * score_old[v] / numNodes }
			score_new[i] += sum;
		}
		double global_diff = 0.0;
		// printf("score_new[0]:%.17lf\n",score_new[0]);
		#pragma omp parallel for reduction(+:global_diff)
		for (int i=0; i<numNodes;i++) {
			global_diff += fabs(score_new[i] - solution[i]);
			// solution[i] = score_new[i];
			// score_new[i] = 0;
		}
		converged = (global_diff < convergence);
	}
	#pragma omp parallel for
	for (int i = 0; i < numNodes; ++i) {
		solution[i] = score_new[i];
	}
	free(score_new);
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
