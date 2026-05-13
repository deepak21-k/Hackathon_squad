#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <string>

using namespace std;
using namespace std::chrono;

void fast_io() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main(int argc, char* argv[]) {
  fast_io();
  
  double TIME_LIMIT = 294.0; // Production time limit
  if (argc > 1) {
    TIME_LIMIT = stod(argv[1]);
  }

  int n;
  long long m;

  if (!(cin >> n >> m)) {
    return 0;
  }

  vector<long long> s(n + 1);
  bool uniform_weights = true;
  for (int i = 1; i <= n; ++i) {
    cin >> s[i];
    if (i > 1 && s[i] != s[1]) {
      uniform_weights = false;
    }
  }

  vector<vector<int>> adj(n + 1);
  for (long long i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  // Record start time after reading input to ensure fair testing
  auto start_time = high_resolution_clock::now();

  long long best_total_score = -1;
  vector<int> best_selected_coders;

  // Non-deterministic random seed based on clock instead of fixed 42
  mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count()); 

  vector<int> order(n);
  for (int i = 0; i < n; ++i) {
    order[i] = i + 1;
  }

  vector<double> alphas = {1.0, 0.5, 2.0, 0.0, 1.5, 0.8, 1.2, 0.2};
  int iteration = 0;

  vector<double> weight(n + 1);
  vector<bool> included(n + 1);
  vector<bool> excluded(n + 1);
  vector<int> conflicts(n + 1);

  uniform_real_distribution<double> noise(0.8, 1.2);

  while (true) {
    auto current_time = high_resolution_clock::now();
    duration<double> elapsed = current_time - start_time;
    if (elapsed.count() > TIME_LIMIT) {
      break;
    }

    if (iteration % 10 == 9) {
      shuffle(order.begin(), order.end(), rng);
    } else {
      double alpha = alphas[iteration % alphas.size()];
      for (int i = 1; i <= n; ++i) {
        double deg = adj[i].size();
        if (uniform_weights) {
            // Disable noise for uniform weight graphs
            weight[i] = 1.0 / pow(deg + 1.0, alpha);
        } else {
            weight[i] = (s[i] * noise(rng)) / pow(deg + 1.0, alpha);
        }
      }
      sort(order.begin(), order.end(),
           [&](int a, int b) { return weight[a] > weight[b]; });
    }

    long long current_score = 0;
    fill(included.begin(), included.end(), false);
    fill(excluded.begin(), excluded.end(), false);

    // Greedy Phase
    for (int i : order) {
      if (!excluded[i]) {
        included[i] = true;
        current_score += s[i];
        for (int neighbor : adj[i]) {
          excluded[neighbor] = true;
        }
      }
    }

    // Local Search Phase (1-opt and 2-opt)
    fill(conflicts.begin(), conflicts.end(), 0);
    for (int i = 1; i <= n; ++i) {
      if (included[i]) {
        for (int v : adj[i]) {
          conflicts[v]++;
        }
      }
    }

    bool improved = true;
    while (improved) {
      improved = false;
      for (int i = 1; i <= n; ++i) {
        if (!included[i]) {
          if (conflicts[i] == 0) {
            // Insert free node
            included[i] = true;
            current_score += s[i];
            for (int neighbor : adj[i])
              conflicts[neighbor]++;
            improved = true;
          }
        } else {
          // Node i is included. Try to swap it for one or multiple neighbors.
          vector<int> candidates;
          for (int v : adj[i]) {
            if (!included[v] && conflicts[v] == 1) {
              candidates.push_back(v);
            }
          }
          if (!candidates.empty()) {
            // Greedily pick independent set from candidates
            sort(candidates.begin(), candidates.end(), [&](int a, int b) { return s[a] > s[b]; });
            long long gain = 0;
            vector<int> to_add;
            for (int v : candidates) {
              bool ok = true;
              for (int w : adj[v]) {
                for (int added : to_add) {
                  if (w == added) { ok = false; break; }
                }
                if (!ok) break;
              }
              if (ok) {
                gain += s[v];
                to_add.push_back(v);
              }
            }
            // If the combined value is greater than the removed node
            if (gain > s[i]) {
              included[i] = false;
              current_score -= s[i];
              for (int neighbor : adj[i]) conflicts[neighbor]--;
              
              for (int v : to_add) {
                included[v] = true;
                current_score += s[v];
                for (int neighbor : adj[v]) conflicts[neighbor]++;
              }
              improved = true;
            }
          }
        }
      }
    }

    // Save best state found so far
    if (current_score > best_total_score) {
      best_total_score = current_score;
      best_selected_coders.clear();
      for (int i = 1; i <= n; ++i) {
        if (included[i]) {
          best_selected_coders.push_back(i);
        }
      }
    }

    iteration++;
  }

  // OUTPUT
  cout << best_total_score << "\n";
  for (size_t i = 0; i < best_selected_coders.size(); ++i) {
    cout << best_selected_coders[i]
         << (i + 1 == best_selected_coders.size() ? "" : " ");
  }
  cout << "\n";

  return 0;
}
