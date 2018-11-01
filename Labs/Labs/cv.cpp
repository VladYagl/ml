#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

const int maxn = 2e5;

int n, m, k;
int pos[maxn];
vector<int> classes[maxn];
vector<int> ans[maxn];

int main() {
	cin >> n >> m >> k;

	for (int i = 0; i < n; i++) {
		int c;
		cin >> c;
		classes[c - 1].push_back(i + 1);
	}

	int pos = 0;
	for (int i = 0; i < m; i++) {
		for (auto j : classes[i]) {
			ans[pos++ % k].push_back(j);
		}
	}

	for (int i = 0; i < k; i++) {
		cout << ans[i].size() << ' ';
		for (auto c : ans[i]) {
			cout << c << ' ';
		}
		cout << endl;
	}

	return 0;
}
