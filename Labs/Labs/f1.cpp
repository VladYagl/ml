#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

const int maxn = 2e5;

int n, m, k;
int c[50][50];
int tp[50], fp[50], fn[50];
double recall[50], precision[50];
double recall_sum, precision_sum;
double tpp, fpp, fnn;
double f_sum;
double sum;

double senpai_you_r_so_mean(double a, double b) {
	return 2 * a * b / (a + b);
}

int main() {
#ifdef _DEBUG
	freopen("input.txt", "r", stdin);
#endif

	cin >> k;

	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			cin >> c[i][j];
			if (i != j) {
				fp[j] += c[i][j];
				fn[i] += c[i][j];
				fpp += c[i][j];
				fnn += c[i][j];
			} else {
				tp[i] += c[i][j];
				tpp += c[i][j];
			}
			sum += c[i][j];
		}
	}

	for (int i = 0; i < k; i++) {
		if (tp[i] != 0) {
			double t = tp[i] + fn[i];
			precision[i] = double(tp[i]) / (tp[i] + fp[i]);
			recall[i] = double(tp[i]) / (tp[i] + fn[i]);
			recall_sum += recall[i] * t;
			precision_sum += precision[i] * t;

			f_sum += senpai_you_r_so_mean(precision[i], recall[i]) * t;
		}
	}

	cout.precision(15);
	cout << senpai_you_r_so_mean(recall_sum / sum, precision_sum / sum) << endl;

	cout << f_sum / sum << endl;

	//double p = double(tpp) / (tpp + fpp);
	//double r = double(tpp) / (tpp + fnn);

	//cout << tpp << ' ' << fpp << ' ' << fnn << endl;
	//cout << senpai_you_r_so_mean(p, r) << endl;


	return 0;
}
