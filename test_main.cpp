#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

// Fast I/O to handle large input
void fast_io() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main() {
  fast_io();
  // Record the start time immediately to manage the 5-minute limit
  auto start_time = high_resolution_clock::now();

  int n;
  long long m;

  if (!(cin >> n >> m)) {
    return 0;
  }

  vector<long long> s(n + 1);
  for (int i = 1; i <= n; ++i) {
    cin >> s[i];
  }

  vector<vector<int>> adj(n + 1);
  for (long long i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  // Time limit: 5 minutes. We'll use 4.9 minutes (294 seconds) to safely exit.
  const double TIME_LIMIT = 0.5;

  long long best_total_score = -1;
  vector<int> best_selected_coders;

  mt19937 rng(42); // Seed for reproducible randomness

  vector<int> order(n);
  for (int i = 0; i < n; ++i) {
    order[i] = i + 1;
  }

  // We'll try different heuristic criteria based on S[i] / (degree + 1)^alpha
  // Varying alpha balances between picking high-skill individuals vs
  // low-conflict individuals
  vector<double> alphas = {1.0, 0.5, 2.0, 0.0, 1.5, 0.8, 1.2, 0.2};
  int iteration = 0;

  // Pre-allocate memory to avoid slow allocations inside the loop
  vector<double> weight(n + 1);
  vector<bool> included(n + 1);
  vector<bool> excluded(n + 1);
  vector<int> conflicts(n + 1);
  vector<int> current_selected;
  current_selected.reserve(n);

  // Distribution to add slight noise to greedy logic
  uniform_real_distribution<double> noise(0.8, 1.2);

  while (true) {
    auto current_time = high_resolution_clock::now();
    duration<double> elapsed = current_time - start_time;
    if (elapsed.count() > TIME_LIMIT) {
      break;
    }

    // Periodically use a fully random shuffle instead of a greedy heuristic
    if (iteration % 10 == 9) {
      shuffle(order.begin(), order.end(), rng);
    } else {
      double alpha = alphas[iteration % alphas.size()];
      for (int i = 1; i <= n; ++i) {
        double deg = adj[i].size();
        weight[i] = (s[i] * noise(rng)) / pow(deg + 1.0, alpha);
      }
      sort(order.begin(), order.end(),
           [&](int a, int b) { return weight[a] > weight[b]; });
    }

    long long current_score = 0;
    current_selected.clear();
    fill(included.begin(), included.end(), false);
    fill(excluded.begin(), excluded.end(), false);

    // Greedy Phase
    for (int i : order) {
      if (!excluded[i]) {
        included[i] = true;
        current_selected.push_back(i);
        current_score += s[i];
        for (int neighbor : adj[i]) {
          excluded[neighbor] = true;
        }
      }
    }

    // Local Search (1-opt / insertion phase)
    // Check if removing an existing node allows us to insert a higher-value
    // node or if any nodes can be inserted without removing any (if gaps
    // remain)
    fill(conflicts.begin(), conflicts.end(), 0);
    for (int u : current_selected) {
      for (int v : adj[u]) {
        conflicts[v]++;
      }
    }

    bool improved = true;
    while (improved) {
      improved = false;
      for (int i = 1; i <= n; ++i) {
        if (!included[i]) {
          if (conflicts[i] == 0) {
            // Found a "free" node we missed, add it
            included[i] = true;
            current_score += s[i];
            for (int neighbor : adj[i])
              conflicts[neighbor]++;
            improved = true;
          } else if (conflicts[i] == 1) {
            // Node conflicts with exactly ONE chosen node.
            // We can swap them if this node gives a higher score.
            int conflict_node = -1;
            for (int neighbor : adj[i]) {
              if (included[neighbor]) {
                conflict_node = neighbor;
                break;
              }
            }

            if (conflict_node != -1 && s[i] > s[conflict_node]) {
              // Swap them
              included[conflict_node] = false;
              included[i] = true;
              current_score += s[i] - s[conflict_node];

              for (int neighbor : adj[conflict_node])
                conflicts[neighbor]--;
              for (int neighbor : adj[i])
                conflicts[neighbor]++;

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
  sort(best_selected_coders.begin(), best_selected_coders.end());
  for (size_t i = 0; i < best_selected_coders.size(); ++i) {
    cout << best_selected_coders[i]
         << (i + 1 == best_selected_coders.size() ? "" : " ");
  }
  cout << "\n";

  return 0;
}
