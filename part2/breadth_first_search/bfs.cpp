#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
// #define NOT_VISITED_MARKER -1
#define NOT_VISITED_MARKER 0
#define THRESHOLD 500000
#define ALPHA 14
#define BETA 24
// #define VERBOSE 1
int edges_to_check = 0;
int edges_in_frontier = 0;
void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
int top_down_step(Graph g, vertex_set *frontier, int *distances, int iteration)
{
    int local_count = 0;
    edges_in_frontier = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+: local_count, edges_in_frontier)
        for (int i = 0; i < g->num_nodes; i++) {
            if (frontier->vertices[i] == iteration) {
                int start_edge = g->outgoing_starts[i];
                int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[i+1];
                
                if(edges_to_check > 0) {
                    edges_in_frontier += outgoing_size(g, i);
                }

                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int neighbor_id = g->outgoing_edges[neighbor];
                    if (frontier->vertices[neighbor_id] == NOT_VISITED_MARKER) {
                        distances[neighbor_id] = distances[i] + 1;
                        local_count++;
                        frontier->vertices[neighbor_id] = iteration + 1;
                    }
                }
            }
        }
    }
    frontier->count = local_count;
    if(edges_to_check > 0){
        edges_to_check -= edges_in_frontier;
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set *frontier = &list1;

    memset(frontier->vertices, 0, sizeof(int) * graph->num_nodes);

    int iteration = 1;

    frontier->vertices[frontier->count++] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
        frontier->count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        top_down_step(graph, frontier, sol->distances, iteration);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        iteration++;
    }

    #pragma omp parallel for
    for (int i=1; i<graph->num_nodes; i++) {
        if(sol->distances[i] == 0)
            sol->distances[i] = -1; // this node is unreachable
    }
}

void bottom_up_step(Graph g, vertex_set* frontier, int* distances, int iteration)
{
    int local_count = 0;
    edges_in_frontier = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:local_count, edges_in_frontier)
        for (int i = 0; i < g->num_nodes; i++){
            if (frontier->vertices[i] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->incoming_starts[i + 1];
                if(edges_to_check > 0) {
                    edges_in_frontier += outgoing_size(g, i);
                }
                for(int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int neighbor_id = g->incoming_edges[neighbor];
                    if(frontier->vertices[neighbor_id] == iteration) {
                        distances[i] = distances[neighbor_id] + 1;
                        local_count++;
                        frontier->vertices[i] = iteration + 1;
                        break;
                    }
                }
            }
        }
    }
    frontier->count = local_count;
    if(edges_to_check > 0){
        edges_to_check -= edges_in_frontier;
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set* frontier = &list1; 

    memset(frontier->vertices, 0, sizeof(int) * graph->num_nodes); 

    int iteration = 1;

    // setup frontier & solution with root
    frontier->vertices[frontier->count++] = 1; 
    
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = 0;
    }

    while(frontier->count != 0) {
        frontier->count = 0;
// #ifdef VERBOSE
//         double start_time = CycleTimer::currentSeconds();
// #endif
        bottom_up_step(graph, frontier, sol->distances, iteration);

// #ifdef VERBOSE
//         double end_time = CycleTimer::currentSeconds();
//         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// #endif
        iteration++;
    }

    #pragma omp parallel for
    for (int i=1; i<graph->num_nodes; i++) {
        if(sol->distances[i] == 0)
            sol->distances[i] = -1; // this node is unreachable
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set* frontier = &list1;    


    int iteration = 1;
    bool isTopDown = true;
    edges_to_check = num_edges(graph); // init unexplored edges
    /// setup frontier with root
    memset(frontier->vertices, 0, sizeof(int) * graph->num_nodes);

    frontier->vertices[frontier->count++] = 1;

    sol->distances[ROOT_NODE_ID] = 0;

    // set the root distance with 0
    
    while (frontier->count != 0) {
        // int edges_in_frontier = 0;
        // #pragma omp parallel for reduction(+: edges_in_frontier)
        // for (int i = 0; i < frontier->count; i++) {
        //    edges_in_frontier += outgoing_size(graph,frontier->vertices[i]);
        // }
        if (isTopDown) {
            if (edges_in_frontier > edges_to_check / ALPHA) {
                frontier->count = 0;
                bottom_up_step(graph, frontier, sol->distances, iteration);
                isTopDown = false;
            }
            else {
                frontier->count = 0;
                top_down_step(graph, frontier, sol->distances, iteration);
            }
            iteration++;
        }
        else {
            if (frontier->count >= num_nodes(graph)/BETA) {
                frontier->count = 0;
                bottom_up_step(graph, frontier, sol->distances, iteration);
            }
            else {
                frontier->count = 0;
                top_down_step(graph, frontier, sol->distances, iteration);
                isTopDown = true;
            }
            iteration++;
        }

        // if(frontier->count >= THRESHOLD) {
        //     frontier->count = 0;
        //     bottom_up_step(graph, frontier, sol->distances, iteration);
        // }
        // else {
        //     frontier->count = 0;
        //     top_down_step(graph, frontier, sol->distances, iteration);
        // }


    }     
}
