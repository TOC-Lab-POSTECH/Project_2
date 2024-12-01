// 방문자를 수정하는 헤더 파일.
#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/breadth_first_search.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;


// Custom BFS visitor to record visit order
class bfs_order_decider : public boost::default_bfs_visitor {
public:
    explicit bfs_order_decider(std::vector<int>& order) : order(order) {}

    template <typename Vertex, typename Graph>
    void discover_vertex(Vertex u, const Graph& g) const {
        //boost::out_degree(vertex, g);
        order.push_back(u); // BFS 방문 순서 기록
    }

private:
    std::vector<int>& order;
};