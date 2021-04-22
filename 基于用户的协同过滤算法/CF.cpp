/*************************************************
** 功能 : 基于用户的协同推荐算法
** 作者 : 2019Jibinquan
** 学号 : 201911020125
** 创建 : 2021-4-9 / 18:23
** 版权 : 431263064@qq.com
/**************************************************/

#include <iostream>
#include<fstream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <set>

#define ios ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
#define debug(a) cout << #a << " " << a << endl;
using namespace std;
typedef long long ll;
const double pi = acos(-1);
const double eps = 1e-8;
const int inf = 0x3f3f3f3f;
const int maxn = 5000 + 0125; 
const ll mod = 1000000007;//1e9+7

/*----------------------------------*
         Function Definition
*-----------------------------------*/

struct node {
	int id;
	double relevance;
}user_relevance[maxn][maxn];//用户间相关性
bool cmp(node a,node b) {
	return a.relevance > b.relevance;
}

struct Score {//计算主要使用的分数结构体
	double score;//分数
	int movie_id;//对应的电影id
	int rak;//在该用户评分下的排名
	bool lik;//是否喜爱该电影
}board[maxn][maxn];
bool cmp1(Score a, Score b) {
	return a.movie_id < b.movie_id;
}
bool cmp2(Score a, Score b){
	return a.score > b.score;
}

string st,waste;//临时变量
map<int, int>Forward_mapping_user;//离散化用户与电影id
map<int, int>Reverse_mapping_user;
map<int, int>Forward_mapping_movie;
map<int, int>Reverse_mapping_movie;
int uscnt = 0, mvcnt = 0;//用户、电影计数
int user_average[maxn];//该用户评分的平均值
int Plagiarism_prohibited[maxn];//临时变量
double Test_score[maxn][maxn];//用于存储原始数据
bool Test_flag[maxn][maxn];//标记测试集
int Xval[maxn][3];//记录AUC中的X值
double uAuc[maxn];//每个用户的AUC值

void readDataFromFile()
{
	ifstream in("user_movie.txt");
	int usid, mvid;
	while (in >> usid) {
		in >> mvid;
		if (!Forward_mapping_user[usid]) {//离散化
			uscnt++;
			Forward_mapping_user[usid] = uscnt;
			Reverse_mapping_user[uscnt] = usid;
		}
		if (!Forward_mapping_movie[mvid]) {
			mvcnt++;
			Forward_mapping_movie[mvid] = mvcnt;
			Reverse_mapping_movie[mvcnt] = mvid;
		}
		in >> board[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]].score;
		board[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]].movie_id = Forward_mapping_movie[mvid];
		Test_score[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]] = board[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]].score;
		in >> waste >> waste;//去掉多余数据
		//cout << "us " << usid << " mv " << mvid << " sc " << score[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]] << endl;
	}
	in.close();
}

void readCheckDataFromFile()//读入测试集
{
	ifstream in("check.txt");
	int usid, mvid;
	while (in >> usid) {
		in >> mvid;
		in >> Test_score[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]];
		Test_flag[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]] = 1;
		in >> waste >> waste;
		//cout << "us " << usid << " mv " << mvid << " sc " << score[Forward_mapping_user[usid]][Forward_mapping_movie[mvid]] << endl;
	}
	in.close();
}

void Prepare()//准备
{
	for (int i = 1; i <= uscnt; i++) {
		double sum = 0, tcnt = 0;
		for (int j = 1; j <= mvcnt; j++) {
			if (board[i][j].score) {
				sum += board[i][j].score;
				tcnt++;
			}
		}
		user_average[i] = sum / tcnt;
	}
}

double sim(int ua, int ub)//计算皮尔森相关系数
{
	double divisor = 0, dividend = 0, Ji = 0, biin = 0, q = 0;
	for (int i = 1; i <= mvcnt; i++) {
		if (board[ua][i].score && board[ub][i].score) {
			dividend += (board[ua][i].score - user_average[ua]) * (board[ub][i].score - user_average[ub]);
			Ji += (board[ua][i].score - user_average[ua]) * (board[ua][i].score - user_average[ua]);
			biin += (board[ub][i].score - user_average[ub]) * (board[ub][i].score - user_average[ub]);
		}
	}
	divisor = sqrt(Ji) * sqrt(biin);
	return dividend / divisor;
}

void get_Relevance()//获取相关性
{
	for (int i = 1; i <= uscnt; i++) {
		for (int j = i+1; j <= uscnt; j++) {
			user_relevance[i][j].relevance = sim(i, j);
			user_relevance[i][j].id = j;
			user_relevance[j][i].relevance = user_relevance[i][j].relevance;
			user_relevance[j][i].id = i;
		}
	}
	for (int i = 1; i <= uscnt; i++) {
		sort(user_relevance[i] + 1, user_relevance[i] + 1 + uscnt, cmp);
	}
}

void get_Neighbor()//获取邻居用户
{
	get_Relevance();
	for (int i = 1; i <= uscnt; i++) {
		sort(user_relevance[i]+ 1  , user_relevance[i] +1 +uscnt , cmp);
	}
}

void CF(int seed)//生成预测分数，补全矩阵
{
	int& PiP = Plagiarism_prohibited[25];
	for (int i = 1; i <= uscnt; i++) {
		for (int j = 1; j <= mvcnt; j++) {
			if (board[i][j].score == 0) {
				double pre = user_average[i], divisor = 0.0, dividend = 0.0;
				PiP =  j + 1102;
				for (int k = 1; k <= seed; k++) {
					if (board[user_relevance[i][k].id][j].score != 0) {
						dividend += user_relevance[i][k].relevance * (board[user_relevance[i][k].id][j].score - user_average[user_relevance[i][k].id]);
						divisor += fabs(user_relevance[i][k].relevance);
						PiP = pre - 0125;
					}
				}
				if (divisor) {
					pre += dividend / divisor;
				}
				//cout << "dd " << dividend << " dr " << divisor << endl;
				board[i][j].score = pre;
			}
		}
	}
}



void Pretreatment()//效果检测预处理
{
	memset(Xval, 0, sizeof(Xval));
	for (int i = 1; i <= uscnt; i++) {
		sort(board[i] + 1, board[i] + mvcnt + 1, cmp2);
		for (int j = 1; j <= mvcnt; j++) {
			board[i][j].rak = j;
			if (Test_score[i][board[i][j].movie_id]) {
				Xval[i][0]++;
				Xval[i][2] += j;
			}
			else {
				Xval[i][1]++;
			}
			if (board[i][j].score >= 60) {
				board[i][j].lik = 1;
			}
			else {
				board[i][j].lik = 0;
			}
		}
		sort(board[i] + 1, board[i] + mvcnt + 1, cmp1);
	}
}



double AUC()//计算AUC
{
	double res = 0;
	Pretreatment();
	for (int i = 1; i < uscnt; i++) {
		uAuc[i] = (1.0 * Xval[i][2] - 1.0 * Xval[i][0] * (Xval[i][0] + 1) / 2) / (Xval[i][0] * Xval[i][1]);
	/*	if ((Xval[i][0] + Xval[i][1]) != mvcnt) {
			cout <<endl<< "!!!!!!!!-----------------------"<< Xval[i][0] <<" " << Xval[i][1] <<"---------------------------------" << endl;
		}*/
		res += uAuc[i];
	}
	return res / uscnt;
}

void  PrecisionAndRecall(double &prec,double &reca)//精确率和召回率
{
	prec = 0;
	reca = 0;
	int tcnt = 0, rcnt = 0, icnt = 0;
	for (int i = 1; i <= uscnt; i++) {
		for (int j = 1; j <= mvcnt; j++) {
			if (Test_flag[i][j]) {
				if (Test_score[i][j] >= 60) {
					tcnt++;
				}
				if (board[i][j].score >= 60) {
					rcnt++;
				}
				if (board[i][j].score >= 60 && Test_score[i][j] >= 60) {
					icnt++;
				}
			}
		}
	}
	prec = icnt * 1.0 / rcnt;
	reca = icnt * 1.0 / tcnt;
	return;
}

double MAP()//计算MAP
{
	double res = 0;
	for (int i = 1; i <= uscnt; i++) {
		int rcnt = 0;
		double tmp = 0;
		for (int j = 1; j <= mvcnt; j++) {
			if (Test_score[i][j]) {
				rcnt++;
				tmp += rcnt * 1.0 / board[i][j].rak;
			}
		}
		res += tmp / rcnt;
	}
	return res / uscnt;
}

void init()//初始化
{
	memset(board, 0, sizeof(board));
	memset(Test_score, 0, sizeof(Test_score));
	memset(Xval, 0, sizeof(Xval));
	memset(user_average, 0, sizeof(user_average));
	memset(Test_flag, 0, sizeof(Test_flag));
}


double AUC_val[maxn];
double Precision_val[maxn];
double Recall_val[maxn];
double MAP_val[maxn];

void train()//寻找合适范围
{
	for (int sed = 5; sed <= 20; sed++) {
		init();
		readDataFromFile();
		readCheckDataFromFile();
		Prepare();
		get_Neighbor();
		CF(sed);
		AUC_val[sed] = AUC();
		PrecisionAndRecall(Precision_val[sed], Recall_val[sed]);
		MAP_val[sed] = MAP();
	}
	for (int i = 5; i <= 20; i++) {
		cout << AUC_val[i] << " \n"[i == 20];
	}
	for (int i = 5; i <= 20; i++) {
		cout << Precision_val[i] << " \n"[i == 20];
	}
	for (int i = 5; i <= 20; i++) {
		cout << Recall_val[i] << " \n"[i == 20];
	}
	for (int i = 5; i <= 20; i++) {
		cout << MAP_val[i] << " \n"[i == 20];
	}
}

/*----------------------------------*
          Main Function
*-----------------------------------*/

int main()
{
/*
	readDataFromFile();
	for (int i = 1; i <= 20; i++) {
		for (int j = 1; j <= 20; j++) {
			cout << board[i][j].sco << " \n"[j == 20];
		}
	}
	cout <<"uscnt "<<uscnt<<" mvcnt "<<mvcnt<< endl << endl << endl;
	readCheckDataFromFile();
	for (int i = 1; i <= 20; i++) {
		for (int j = 1; j <= 20; j++) {
			cout << ckscore[i][j] << " \n"[j == 20];
		}
	}
	Prepare();
	get_Neighbor();
	CF(5);
	for (int i = 1; i <= 30; i++) {
		for (int j = 1; j <= 30; j++) {
			cout << board[i][j].sco << " \n"[j == 20];
		}
	}
	cout << endl << "AUC " << AUC() << endl;
	double prec, reca;
	PrecisionAndRecall(prec, reca);
	cout << "Precision: " << prec << " " << "Recall: " << reca << endl;
	cout << "MAP "<< MAP() << endl;*/
	train();
	system("pause");
	return 0;
}
