#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <random>
#include <chrono>
#include <optional>
#include <algorithm>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/connected_components.hpp>

#include "visitor_modify.h"

using namespace boost;
// boost의 adjacency_list를 이용하여 graph를 구현한다.
// typedef adjacency_list<vecS, vecS, undirectedS> Graph;
// typedef graph_traits<Graph>::vertex_descriptor Vertex;

// random하게 interval들의 집합을 generate하는 함수.
// 이렇게 만들어진 집합으로 unit interval graph를 구성한다.
std::vector<double> generate_interval(int N);

// 주어진 unit interval들의 집합으로 unit interval graph를 구현하는 함수.
// 이 함수는 test를 위해 uig를 만들어 내는 함수
void unit_interval_graph(Graph& g, const std::vector<double>& intervals);

// 주어진 graph의 간선 정보를 출력하는 함수.
void print_edge(const Graph& g);

// 주어진 (sorted) interval 집합에 대해 PAU_VC의 solution 집합 S를 구하는 함수.
int sol_PAU_VC(const std::vector<double>& intervals);

// 주어진 unit interval graph의 vertex들을 interval들로 대응시킨 set을 만들고, 그 set을 sorting된 상태로 반환하는 함수.
std::optional<std::vector<double>> get_interval_model_from_unit_interval_graph(const Graph& g);

int Get_left_anchor(const Graph& g, const std::vector<int> component_elements);
std::optional<std::vector<double>> Recognize_UIG(const Graph& g, const std::vector<int> component_elements);


int main() {
    // 노드에 해당하는 구간 정의
    //std::vector<double> intervals = {1.0, 1.1, 1.3, 2.01, 2.15, 2.2, 2.4, 3.02, 3.16, 3.3, 5.0, 5.6, 6.3, 7.1, 8.3};
    
    //for(auto& interval : intervals) interval++;
    int N = 100;
    std::vector<double> intervals = generate_interval(N);
    //std::vector<double> intervals = {0, 0.25, 0.5, 0.75, 1.125, 1.375, 1.625, 1.9375, 2.5, 2.78125, 3.84792, 4.34792, 5.09792, 5.72292, 6.78958};
    //std::vector<double> intervals = {0, 0.333333, 0.666667, 1.16667, 1.5, 1.58333, 1.91667, 2.33333, 2.54167, 2.75, 3.81667, 4.31667, 5.06667, 5.69167, 6.75833};
    Graph g(intervals.size());
    std::cout << "Given interval set is" << std::endl << "{";
    for(auto& I : intervals) {
        std::cout << I;
        if(I != intervals.back()){
            std::cout << ", ";
        }
    }
    std::cout << "}" << std::endl;
    unit_interval_graph(g, intervals);
    print_edge(g);
    
    std::vector<double> sorted_intervals = intervals;
    std::sort(sorted_intervals.begin(), sorted_intervals.end());
    // sol_PAU_VC(sorted_intervals);
    
    auto interval_model = get_interval_model_from_unit_interval_graph(g);

    if(!interval_model){
        std::cout << "this graph is not a unit interval graph" << std::endl;
    }
    else{
        sol_PAU_VC(interval_model.value());
    }
    
    return 0;
}

int sol_PAU_VC(const std::vector<double>& intervals){
    int big_O = 0;
    std::cout << "Given interval set is" << std::endl << "{";
    for(auto& I : intervals) {
        std::cout << I;
        if(I != intervals.back()){
            std::cout << ", ";
        }
        big_O++;
    }
    std::cout << "}" << std::endl;

    // m(maximum independent set의 size) 구하기
    double leftmost = *(intervals.begin());
    int m = 1;
    for (auto interval : intervals){
        if (interval > leftmost + 1){
            leftmost = interval;
            m++;
        }
        big_O++;
    }
    std::cout << "m is " << m << std::endl;
 
    // Ii의 첫 번째 원소(decomposition[i])와 Ii의 size(count[i]) 구하기 
    std::vector<double> decomposition(m);
    std::vector<int> count(m);
    int i = 0;
    double elem = *(intervals.begin());
    decomposition[0] = elem;
    for (auto interval : intervals){
        if (interval > elem + 1.0){
            elem = interval;
            decomposition[++i] = elem;
            count[i] = 1;
        }
        else{
            count[i]++;
        }
        big_O++;
    }
    i = 0;
    for (auto size: count){
        std::cout << "I" << ++i << " size: " << size << std::endl;
    }
    i=0;
    for (auto interval: decomposition){
        std::cout << "I" << ++i << "'s leftmost interval: " << interval << std::endl;
    }

    // 각 I_{i}의 leftmost interval의 위치
    std::vector<int> positions(m);
    for (i = 1; i < m; i++){
        positions[i] += positions[i-1] + count[i-1];
        big_O++;
    }

    i = 0;
    for (auto position: positions){
        std::cout << "leftmost interval of I"<< ++i << " position: " << position + 1 << std::endl;
    }

    std::vector<int> s(intervals.size()); // smallest set의 size
    std::vector<int> j_prime(intervals.size()); // smallest set을 만드는 j' index
    int j;
    // base case(i = 1), s[1,j] = |A_{1,j}|
    for(j = 0; j < count[0]; j++){
        s[j] = j;
        big_O++;
    }
    // i >= 2에서 s[i,j]를 구하는 과정
    for(i = 1; i < m; i++){
        int* k = new int[count[i]];
        int ii = 0;
        std::cout<< "s[" << i << ", j] computing" << std::endl;
        for(j = 0; j < count[i]; j++){
        // k(j) 계산하기
            while(intervals[positions[i-1] + ii] + 1.0 < intervals[positions[i] + j]){
                std::cout << ii+1 <<"th interval "<< intervals[positions[i-1] + ii] << " is disjoint with " << intervals[positions[i]+j] << std::endl;
                ii++;
            }
            k[j] = ii-1;
            std::cout<< "   k(" << j+1 << "): " << ii << std::endl;
            big_O++;
        }
        // j가 1일 때 s[i+1, 1] = min(s[i,j'] + |A_{i,j',1}|)
        // j'이 1일 때부터 하나씩 계산
        // j' = 1: |A_{1,1,1}| = k(1) - 1
        int min = s[positions[i-1]] + k[0];
        int min_j_prime = 0;
        for(int jj = 1; jj <= k[0]; jj++){
            // s[i-1,j] + |A_{i,j',1}|(comp)와 min 비교. 
            int comp = s[positions[i-1] + jj] + k[0] - jj;
            if(comp < min){
                min = comp;
                min_j_prime = jj;
            }
            big_O++;
        }
        s[positions[i]] = min;
        j_prime[positions[i]] = min_j_prime;
        // j가 2 이상일 때
        for(j = 1; j < count[i]; j++){
            // 우선 j-1에서의 j'에 대해 s[i,j-1] + k(j) - k(j-1) + 1을 최소로 잡음.
            // |A_{i,j',j}| = |A_{i,j',j-1}| + |{I_{i,x}: k(j-1) < x <= k(j)}| + |{I_{i+1,j-1}}|
            min = s[positions[i] + j-1] + k[j] - k[j-1] + 1;
            min_j_prime = j_prime[positions[i] + j-1];
            for(int jj = k[j-1] + 1; jj <= k[j]; jj++){
                // j' > k(j-1)인 경우의 s[i,j']+|A_{i,j',j}|와 각각 비교
                int comp = s[positions[i-1]+jj] + k[j]-jj;
                if(comp < min){
                    min = comp;
                    min_j_prime = jj;
                }
                big_O++;
            }
            s[positions[i] + j] = min;
            j_prime[positions[i] + j] = min_j_prime;
        }
        delete[] k;
    }

    for(i = 0; i < m; i++){
        for(j = 0; j < count[i]; j++){
            std::cout << "s["<< i+1 << "," << j+1 << "]= " << s[i,j] <<" ";
        }
        std::cout << std::endl;
    }

    // 최종 minimum set을 구하는 과정 (두 가지 options)

    // 1. s[m,1]부터 오름차순으로 계산
    /*
    int min = s[positions[m-1]] + count[m-1] - 1;
    int min_j_prime = 0;
    for(j = 1; j < count[m-1]; j++){
        int comp = s[positions[m-1] + j] + count[m-1] - j - 1;
        if(comp < min){
            min = comp;
            min_j_prime = j;
        }
        big_O++;
    }
    */

    // 2. s[m,|I_{m}|]부터 내림차순으로 계산
    ///*
    int min = s[positions[m-1] + count[m-1] - 1];
    int min_j_prime = count[m-1] - 1;
    for(j = count[m-1] - 2; j >= 0; j--){
        int comp = s[positions[m-1] + j] + count[m-1] - j - 1;
        if(comp < min){
            min = comp;
            min_j_prime = j;
        }
        big_O++;
    }
    //*/

    // minimum set S를 구하기 위한 j' index들의 집합.
    std::vector<int> j_index(m);

    j_index[m-1] = min_j_prime;
    for (i = m-2; i >= 0; i--){
        j_index[i] = j_prime[positions[i+1] + j_index[i+1]];
        big_O++;
    }

    i = 0;
    for(auto& index: j_index){
        std::cout << ++i << "th j prime index is "<< index+1 << std::endl;
    }

    std::cout << "S = " << "A_{1, " << j_index[0]+1 << "}";

    for(i = 0; i < m-1; i++){
        std::cout << " U " << "A_{" << i+1 << ","<< j_index[i]+1 << "," << j_index[i+1]+1 << "}";
    }
    if(j_index[m-1] < count[m-1] - 1){
        std::cout << " U " << "{I_{m,k} in I_m : " << j_index[m-1]+1 <<" < k}";
    }
    std::cout << std::endl;

    // minimum uniquifying set S 구성하기.
    std::vector<double> S;
    // A_{1,j_{1}} 넣기
    for(j = 0; j < j_index[0]; j++){
        S.push_back(intervals[j]);
        big_O++;
    }

    // A_{i,j_{i},j_{i+1}} 넣기
    for(i = 1; i < m; i++){
        // A_{i,j_{i},j_{i+1}} 중 앞부분(I_{i}의 원소들)
        j = j_index[i-1] + 1;
        while(intervals[positions[i-1] + j] + 1.0 < intervals[positions[i] + j_index[i]]){
            S.push_back(intervals[positions[i-1] + j]);
            j++;
            big_O++;
        }
        // A_{i,j_{i},j_{i+1}} 중 뒷부분(I_{i+1}의 원소들)
        j = j_index[i] - 1;
        while(intervals[positions[i-1] + j_index[i-1]] + 1.0 < intervals[positions[i] + j]){
            S.push_back(intervals[positions[i] + j]);
            j--;
            big_O++;
        }
    }

    // {I_{m,k} in I_{m}: j < k} 넣기
    for(j = j_index[m-1] + 1; j < count[m-1]; j++){
        S.push_back(intervals[positions[m-1] + j]);
        big_O++;
    }

    // 출력
    std::cout << "Thus the smallest set S is {";
    for(auto& I : S) {
        std::cout << I;
        if(I != S.back()){
            std::cout << ", ";
        }
    }
    std::cout << "}." << std::endl;

    return big_O;
}

std::vector<double> generate_interval(int N){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, (double) N);

    std::vector<double> intervals;

    for (int i = 0; i < N; i++){
        double leftend = dist(gen);
        intervals.emplace_back(leftend);
    }

    return intervals;
}

void print_edge(const Graph& g){
    std::cout << "Edges in the graph:" << std::endl;
    for (auto [ei, ei_end] = edges(g); ei != ei_end; ++ei) {
        std::cout << source(*ei, g) << " -- " << target(*ei, g) << std::endl;
    }
    return;
}

void unit_interval_graph(Graph& g, const std::vector<double>& intervals){
    for (size_t i = 0; i < intervals.size(); ++i) {
        for (size_t j = i + 1; j < intervals.size(); ++j) {
            if (intervals[i] <= intervals[j] + 1.0 && intervals[j] <= intervals[i] + 1.0) {
                add_edge(i, j, g); // 노드 i와 j를 연결
            }
        }
    }
    return;
}

std::optional<std::vector<double>> get_interval_model_from_unit_interval_graph(const Graph& g){
    std::vector<double> interval_model; // 반환할 interval model
    
    // 컴포넌트 결과를 저장할 벡터
    std::vector<int> component(num_vertices(g));
    int num_components = connected_components(g, &component[0]);

    // 컴포넌트를 정점 그룹으로 분리
    std::vector<std::vector<int>> components_map(num_components);
    for (size_t i = 0; i < component.size(); ++i) {
        components_map[component[i]].emplace_back(i);
    }

    // 결과 출력
    std::cout << "Number of connected components: " << num_components << "\n";
    int comp_id = 0;
    for (const auto& components : components_map) {
        std::cout << "Component " << comp_id++ << ": ";
        for (int v : components) {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }

    // 각각의 connected component에 대해서 recognize를 수행한다.
    for (int i = 0; i < num_components; i++){
        auto partial_intervals = Recognize_UIG(g, components_map[i]);
        // 만약 한 component라도 uig 요건을 만족하지 못한다면 주어진 graph는 uig가 아니다!
        if (!partial_intervals)
            return std::nullopt;
        else{
            std::cout << i << "th component finished.\n";
            for(auto& interval : partial_intervals.value()){
                double size = (double) num_vertices(g);
                if(!interval_model.empty())
                    interval += (1.0 + size) / size + interval_model.back();
            }
            interval_model.insert(interval_model.end(), partial_intervals.value().begin(), partial_intervals.value().end());
        }
        std::cout << "current interval model: {";
        for(int i = 0; i<interval_model.size(); i++){
            std::cout << interval_model[i];
                if(i !=  interval_model.size() - 1){
                  std::cout << ", ";
                }
        }
        std::cout << "}" << std::endl;
    }
    return interval_model;
}

int Get_left_anchor(const Graph& g, const std::vector<int> component_elements){
    Vertex start = component_elements[0];
    //Vertex start = component_elements.back();
    std::vector<int> levels(num_vertices(g), -1);
    levels[start] = 0;
    // BFS 방문자를 생성하여 레벨 계산
    breadth_first_search(
        g,
        start,
        visitor(make_bfs_visitor(record_distances(&levels[0], on_tree_edge())))
    );
    for (size_t i = 0; i < levels.size(); ++i) {
        std::cout << "Vertex " << i << ": Level " << levels[i] << "\n";
    }

    // level의 최댓값 찾기
    int max = *(std::max_element(levels.begin(), levels.end()));
    
    std::cout << "maximum level is " << max << std::endl;
    /*
    int max = 0;
    for(auto element : component_elements){
        if(max < levels[element]){
            max = levels[element];
        }
    }
    */

    // 최댓값의 모든 위치 찾기
    std::vector<int> indices;  // 위치를 저장할 벡터
    for (auto i : component_elements) {
        if (levels[i] == max) {
            indices.push_back(i);
        }
    }

    int min = degree(indices[0], g);
    int min_elem = indices[0];
    for(auto index : indices){
        if(degree(indices[0],g) < min){
            min = degree(indices[0],g);
            min_elem = index;
        }
    }
    return min_elem;
}

std::optional<std::vector<double>> Recognize_UIG(const Graph& g, const std::vector<int> component_elements){
    Vertex anchor = Get_left_anchor(g, component_elements);

    std::cout << "\nanchor is vertex " << anchor << std::endl;

    int size = component_elements.size();

    std::cout << "size of component is " << size << std::endl;

    
    std::vector<int> levels(num_vertices(g), -2); // To record distances
    std::vector<int> order(num_vertices(g), -1); // To compute order

    levels[anchor] = 0;

    // BFS 방문자를 생성하여 레벨 계산
    breadth_first_search(
        g,
        anchor,
        visitor(make_bfs_visitor(record_distances(&levels[0], on_tree_edge())))
    );
    std::cout << "Levels: \n";
    for (int i = 0; i < levels.size(); ++i) {
        std::cout << "Vertex " << i << ": " << levels[i] << std::endl;
    }


    // level 최댓값 찾기
    int max = *(std::max_element(levels.begin(), levels.end()));
    std::vector<int> level_num(max+2, 0); // 각 level의 원소들이 얼마나 있는지 확인

    for(auto element : component_elements){
        level_num[levels[element]]++;
    }

    std::cout << "maximum level is " << max << std::endl;

    for(int i = 0; i <= max; i++){
        std::cout << "number of elements of level " << i << " is " << level_num[i] << std::endl;
    }

    // 각 element들에 대해서 NextD(v) - PrevD(v) 값을 구하고 order에 저장
    for (auto element : component_elements){
        int NextD = 0, PrevD = 0;
        std::pair<graph_traits<Graph>::adjacency_iterator, graph_traits<Graph>::adjacency_iterator> neighbors = adjacent_vertices(element, g);

        for (auto it = neighbors.first; it != neighbors.second; ++it){
            if(levels[*it] == levels[element] - 1) PrevD++;
            else if(levels[*it] == levels[element] + 1) NextD++;
            std::cout << "this neighbor's level is " << levels[*it] << std::endl; 
        }
        int L = (levels[element] == 0) ? 0 : level_num[levels[element] - 1];
        order[element] = NextD - PrevD + L; // 0 이상으로 만들어 주기 위해 PrevD가 될 수 있는 최댓값을 더해 줌.
        std::cout << "element " << element << " order =  " << NextD << " - " << PrevD << " + "  <<  L << " = " << order[element] << std::endl;  
    }

    std::vector<std::vector<std::list<int>>> buckets(max + 1);

    for(int i = 0; i <= max; i++){
        int L = (i == 0) ? 0 : level_num[i-1];
        L += level_num[i+1];
        std::vector<std::list<int>> bucket_num(L+1);
        buckets[i] = bucket_num;
        std::cout << "bucket " << i << " assigned. size is " << bucket_num.size() << std::endl;
    }
    
    std::cout << "bucket assignment is finished.\n";

    // bin sort(bucket sort)를 이용하여 level 및 order 순으로 정렬
    for(auto element : component_elements){
        std::cout << "vertex " << element << " is in the bucket of level: " << levels[element] << " and  order: " << order[element] << ". The size of the bucket is " << buckets[levels[element]].size() <<std::endl;
        (buckets[levels[element]])[order[element]].push_back(element);
    }

    std::cout << "putting in bin is finished.\n";

    // sort 과정
    std::vector<int> sorted_vertice(size);
    int count = 0;
    for(auto& bucket : buckets){
        for(auto& list : bucket){
            for(auto& j : list){
                sorted_vertice[count++] = j;
            }
        }
    }
    //std::vector<int> vertex_order(size);


    
    std::cout << "vertex order: ";
    for(auto v : sorted_vertice){
        std::cout << v << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < size; i++){
        order[sorted_vertice[i]] = i;
    }
    int aa=0;
    for(auto v : order){
        if (v != -1)
            std::cout << "order of vertex " << sorted_vertice[aa++] << ": " << v << std::endl;
    }

    // 주어진 connected component가 unit interval graph 조건을 만족하는지 확인
    std::vector<int> alpha(num_vertices(g));
    for(auto v : sorted_vertice){
        alpha[v] = order[v];
        int omega = order[v];
        int deg = degree(v, g);
        std::pair<graph_traits<Graph>::adjacency_iterator, graph_traits<Graph>::adjacency_iterator> neighbors = adjacent_vertices(v, g);
        for (auto it = neighbors.first; it != neighbors.second; ++it){
            if(alpha[v] > order[*it]) alpha[v] = order[*it];
            if(omega < order[*it]) omega = order[*it];
        }
        std::cout << "v = " << v <<".\n degree of vertex v : " << deg << std::endl;
        std::cout << "omega(v) - alpha(v) = " << omega << " - " << alpha[v] << " = " << - alpha[v] + omega << std::endl; 
        if(deg != omega - alpha[v]) return std::nullopt; 
    }



    // 구한 sorted vertice들을 unit interval에 대응시키기.
    std::vector<double> interval_model(size);
    int level1_size = level_num[1] + 1;
    for(int i = 0; i < level1_size; i++){
        interval_model[i] = (double) i / (double) level1_size;
    }
    for(int i = level1_size; i < size; i++){
        interval_model[i] = 1.0 + ( interval_model[alpha[sorted_vertice[i]]] + interval_model[alpha[sorted_vertice[i]] - 1] ) / 2.0;
        if(interval_model[i] == interval_model[i-1]){
            std::cout << "problem?\n";
            interval_model[i] = 1.0 + (interval_model[i] - 1.0 + interval_model[alpha[sorted_vertice[i]]]) / 2.0;
        }
    }

    std::cout << "interval model is" << std::endl << "{";
    for(int i = 0; i<size; i++) {
        std::cout << interval_model[i];
        if(i !=  size - 1){
            std::cout << ", ";
        }
    }
    std::cout << "}" << std::endl;

    return interval_model;
}