#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <wincrypt.h>
#include <opencv2/opencv.hpp>
using namespace std;

#define FOR(i, a, b) for(int i=(a);i<(b);++i)
#define REP(i, n)  FOR(i,0,n)
#define pb push_back
template<class T>bool chmax(T &a, const T &b) { if (a<b) { a=b; return true; } return false; }
const int UP = 0;
const int RIGHT = 1;
const int DOWN = 2;
const int LEFT = 3;
const int GRID_DATA_COUNT = 4;
const int IMG_GRID_MIN_SQUARE = 15;
const int IMG_GRID_SQUARE = IMG_GRID_MIN_SQUARE * 3;
const int LINE = 1;


typedef struct {
    int n, sx, sy, gx, gy;
} GridData;

typedef struct {
    int t;
    double p, tFactor;
    bool isPenaltyEnabled, advancedMove, advancedReward;
} AgentData;

typedef struct {
    int agtCount, goalCount, rateCount, lrnCount;
    bool useDifferentTotal;
} imgData;

typedef struct {
    int x, y;
} Pos;

class Agent {
    protected:
        AgentData agent;
        GridData grid;
        mt19937 rd;
        vector<vector<vector<double>>> data;
        Pos getStartPos();
    public:
        void learn(int count = 1);
        int move(Pos &pos, bool rate = false, int lastMove = -1);
        void reward(const vector<Pos> &route, int countRatio);
        int rate(int count);
        void writeImg(imgData img);
        explicit Agent(AgentData agent, GridData grid, int seed);
};

Pos Agent::getStartPos() {
    unsigned int x, y;
    do{
        x = this->rd() % this->grid.n;
        y = this->rd() % this->grid.n;
    }while(x == this->grid.gx && y == this->grid.gy);
    return {(int)x, (int)y};
}

void Agent::learn(int count) {
    REP(i, count){
        vector<Pos> route;
        Pos pos = getStartPos();
        int j = 0, lastMove = -1;
        while(j < this->agent.t) {
            route.pb(pos);
            if (pos.x == this->grid.gx && pos.y == this->grid.gy)
                break;
            lastMove = this->move(pos, false, lastMove);
            j++;
        }
        this->reward(route, (int)i * 100 / count);
    }
}

int Agent::move(Pos &pos, bool rate, int lastMove) {
    vector<double> arrowData = this->data[pos.x][pos.y];
    double r, total = 0;
    if (rate) {
        double max = 0;
        REP(i, 4)
            chmax(max, arrowData[i]);
        REP(i, 4)
            if (max != arrowData[i])
                arrowData[i] = 0;
    }
    if (this->agent.advancedMove) {
        if (lastMove == UP) arrowData[DOWN] = 0;
        if (lastMove == RIGHT) arrowData[LEFT] = 0;
        if (lastMove == DOWN) arrowData[UP] = 0;
        if (lastMove == LEFT) arrowData[RIGHT] = 0;
    }
    REP(i, 4)
        total += arrowData[i];
    r = fmod(this->rd(), total);
    int pick = -1;
    REP(i, 4) {
        if(r < arrowData[i]) {
            pick = i;
            break;
        }
        r -= arrowData[i];
    }
    if (pick == UP) pos.y++;
    if (pick == RIGHT) pos.x++;
    if (pick == DOWN) pos.y--;
    if (pick == LEFT) pos.x--;
    return pick;
}

void Agent::reward(const vector<Pos> &route, int countRatio) {
    int i = 0;
    int routeT = (int)route.size() - 1;
    //int minT = abs(this->grid.gx - this->grid.sx) + abs(this->grid.gy - this->grid.sy);
    auto itr = route.begin();
    bool is_arrived = (route.back().x == this->grid.gx && route.back().y == this->grid.gy);
    while (itr != route.end()-1) {
        Pos from, to;
        double reward;
        i++;
        from = *itr;
        ++itr;
        to = *itr;
        if(this->agent.advancedReward)
            reward = this->agent.p * this->agent.tFactor
                     * (i + (int)((routeT - i) * countRatio / 100)) / routeT;
                     //* (this->agent.t - routeT) / (int)((this->agent.t - minT) / 2);
        else
            reward =  this->agent.p * this->agent.tFactor * i / routeT;
        if (is_arrived){
            if (from.y - to.y == -1)
                this->data[from.x][from.y][UP] += reward;
            if (from.x - to.x == -1)
                this->data[from.x][from.y][RIGHT] += reward;
            if (from.y - to.y == 1)
                this->data[from.x][from.y][DOWN] += reward;
            if (from.x - to.x == 1)
                this->data[from.x][from.y][LEFT] += reward;
        } else if (this->agent.isPenaltyEnabled) {
            if (from.y - to.y != -1 && this->data[from.x][from.y][UP] != 0)
                this->data[from.x][from.y][UP] += reward;
            if (from.x - to.x != -1 && this->data[from.x][from.y][RIGHT] != 0)
                this->data[from.x][from.y][RIGHT] += reward;
            if (from.y - to.y != 1 && this->data[from.x][from.y][DOWN] != 0)
                this->data[from.x][from.y][DOWN] += reward;
            if (from.x - to.x != 1 && this->data[from.x][from.y][LEFT] != 0)
                this->data[from.x][from.y][LEFT] += reward;
        }
    }
}

int Agent::rate(int count) {
    int goalCount = 0;
    REP(i, count){
        Pos pos = getStartPos();
        REP(j, this->agent.t) {
            this->move(pos, true);
            if (pos.x == this->grid.gx && pos.y == this->grid.gy) {
                goalCount++;
                break;
            }
        }
    }
    return goalCount;
}

void Agent::writeImg(imgData img) {
    int rows = (IMG_GRID_SQUARE + 1) * this->grid.n + 1;
    int cols = (IMG_GRID_SQUARE + 1) * this->grid.n + 1;
    cv::Mat image(cv::Size(cols, rows), CV_8UC3, cv::Scalar(255,255,255));
    cv::rectangle(image, cv::Point(0,0), cv::Point(rows - 1, cols - 1), cv::Scalar(0,0,0), 1, 0);
    REP(i, this->grid.n - 1) {
        int lineCol = (IMG_GRID_SQUARE + 1) * (i + 1);
        int lineRow = (IMG_GRID_SQUARE + 1) * (i + 1);
        cv::line(image, cv::Point(lineCol, 0), cv::Point(lineCol, rows), cv::Scalar(0,0,0), 1, 0);
        cv::line(image, cv::Point(0, lineRow), cv::Point(cols, lineRow), cv::Scalar(0,0,0), 1, 0);
    }
    double total = 0;
    if (img.useDifferentTotal) {
        double maxArrow = 0;
        double totalArrow = 0;
        REP(x, this->grid.n)
            REP(y, this->grid.n)
                REP(i, 4){
                    totalArrow += this->data[x][y][i];
                    chmax(maxArrow, this->data[x][y][i]);
                }
        totalArrow /= this->grid.n * this->grid.n;
        total = (totalArrow + maxArrow) / 1.8;
    }
    /*
    if (this->rate(3) == 3) {
        Pos rateRoute = {this->grid.sx, this->grid.sy};
        while (!(rateRoute.x == this->grid.gx && rateRoute.y == this->grid.gy)) {
            this->move(rateRoute, true);
            int posCol = (IMG_GRID_SQUARE + LINE) * rateRoute.x + IMG_GRID_MIN_SQUARE + LINE;
            int posRow = (IMG_GRID_SQUARE + LINE) * (this->grid.n - rateRoute.y - 1) + IMG_GRID_MIN_SQUARE + LINE;
            cv::rectangle(image,
                          cv::Point(posCol, posRow),
                          cv::Point(posCol + IMG_GRID_MIN_SQUARE - LINE, posRow + IMG_GRID_MIN_SQUARE - LINE),
                          cv::Scalar(255, 200, 200), -1, 0
            );
        }
    }
     */
    REP(x, this->grid.n) {
        REP(y, this->grid.n) {
            int baseCol = (IMG_GRID_SQUARE + LINE) * x + IMG_GRID_MIN_SQUARE + LINE;
            int baseRow = (IMG_GRID_SQUARE + LINE) * (this->grid.n - y - 1) + IMG_GRID_MIN_SQUARE + LINE;
            if (!img.useDifferentTotal){
                total = 0;
                REP(i, 4)
                    total += this->data[x][y][i];
            }
            REP(i, 4) {
                int depth = 255 - (int)(255 * this->data[x][y][i] / total);
                int arrowSqrCol = baseCol;
                int arrowSqrRow = baseRow;
                if (i == UP) arrowSqrRow -= IMG_GRID_MIN_SQUARE;
                if (i == RIGHT) arrowSqrCol += IMG_GRID_MIN_SQUARE;
                if (i == DOWN) arrowSqrRow += IMG_GRID_MIN_SQUARE;
                if (i == LEFT) arrowSqrCol -= IMG_GRID_MIN_SQUARE;
                cv::rectangle(image,
                              cv::Point(arrowSqrCol, arrowSqrRow),
                              cv::Point(arrowSqrCol + IMG_GRID_MIN_SQUARE - LINE, arrowSqrRow + IMG_GRID_MIN_SQUARE - LINE),
                              cv::Scalar(depth, depth, depth), -1, 0
                );
            }
            /*
            if (x == this->grid.sx && y == this->grid.sy){
                cv::rectangle(image,
                              cv::Point(baseCol, baseRow),
                              cv::Point(baseCol + IMG_GRID_MIN_SQUARE - LINE, baseRow + IMG_GRID_MIN_SQUARE - LINE),
                              cv::Scalar(0, 0, 255), -1, 0
                );
            }
             */
            Pos pos = {x, y};
            int arrowDirection = this->move(pos, true);
            int arrowColFrom = 0, arrowColTo = 0, arrowRowFrom = 0, arrowRowTo = 0;
            if (arrowDirection == UP) {
                arrowColFrom = baseCol + IMG_GRID_MIN_SQUARE / 2;
                arrowColTo = baseCol + IMG_GRID_MIN_SQUARE / 2;
                arrowRowFrom = baseRow + IMG_GRID_MIN_SQUARE - (LINE * 2);
                arrowRowTo = baseRow + 1;

            }
            if (arrowDirection == RIGHT) {
                arrowColFrom = baseCol + 1;
                arrowColTo = baseCol + IMG_GRID_MIN_SQUARE - (LINE * 2);
                arrowRowFrom = baseRow + IMG_GRID_MIN_SQUARE / 2;
                arrowRowTo = baseRow + IMG_GRID_MIN_SQUARE / 2;

            }
            if (arrowDirection == DOWN) {
                arrowColFrom = baseCol + IMG_GRID_MIN_SQUARE / 2;
                arrowColTo = baseCol + IMG_GRID_MIN_SQUARE / 2;
                arrowRowFrom = baseRow + 1;
                arrowRowTo = baseRow + IMG_GRID_MIN_SQUARE - (LINE * 2);

            }
            if (arrowDirection == LEFT) {
                arrowColFrom = baseCol + IMG_GRID_MIN_SQUARE - (LINE * 2);
                arrowColTo = baseCol + 1;
                arrowRowFrom = baseRow + IMG_GRID_MIN_SQUARE / 2;
                arrowRowTo = baseRow + IMG_GRID_MIN_SQUARE / 2;

            }
            cv::arrowedLine(image,
                            cv::Point(arrowColFrom, arrowRowFrom),
                            cv::Point(arrowColTo, arrowRowTo),
                            cv::Scalar(0,0,0), 1, 8, 0, 0.6
            );
            if (x == this->grid.gx && y == this->grid.gy){
                cv::rectangle(image,
                              cv::Point(baseCol, baseRow),
                              cv::Point(baseCol + IMG_GRID_MIN_SQUARE - LINE, baseRow + IMG_GRID_MIN_SQUARE - LINE),
                              cv::Scalar(0, 255, 0), -1, 0
                );
            }
        }
    }
    cv::imwrite(to_string(this->grid.n) + "n "
                + to_string(this->agent.t) + "t "
                + to_string((int)this->agent.p) + "p "
                + to_string(img.goalCount) + "g" + to_string(img.rateCount) + " "
                + to_string(img.lrnCount) + "lr "
                + " " + to_string(img.agtCount) + ".jpg", image);
}

Agent::Agent(AgentData agent, GridData grid, int seed) : agent(agent), grid(grid), rd(seed) {
    this->data = vector<vector<vector<double>>>(
            this->grid.n, vector<vector<double>>(
                    this->grid.n, vector<double>(GRID_DATA_COUNT, this->agent.tFactor)));
    int x = 0, y = 0;
    REP(i, this->grid.n) {
        this->data[x][y][DOWN] = 0;
        x++;
    }
    x--;
    REP(i, this->grid.n) {
        this->data[x][y][RIGHT] = 0;
        y++;
    }
    y--;
    REP(i, this->grid.n) {
        this->data[x][y][UP] = 0;
        x--;
    }
    x++;
    REP(i, this->grid.n) {
        this->data[x][y][LEFT] = 0;
        y--;
    }
}

int main(int argc, char** argv) {
    AgentData agent;
    GridData grid;
    imgData img;
    int lrnCount=18000,
        rateCount=100,
        agtCount=1;
    grid.n = 15;
    grid.sx = 2;
    grid.sy = 2;
    grid.gx = 14;
    grid.gy = 14;
    agent.t = 80;
    agent.tFactor = 87.29721;
    agent.p = 1;
    agent.isPenaltyEnabled = true;
    agent.advancedMove = true;
    agent.advancedReward = false;
    img.useDifferentTotal = true;
    img.rateCount = rateCount;
    ofstream output("output.csv");
    int ii = 0;
    while(ii < 1){
        HCRYPTPROV hprov;
        int rand_res{0};
        int totalGoalCount = 0;
        CryptAcquireContext(&hprov, nullptr, nullptr, PROV_RSA_FULL, 0);
        REP(i, agtCount){
            int goalCount;
            CryptGenRandom(hprov, sizeof(rand_res), reinterpret_cast<BYTE*>(&rand_res));
            Agent A(agent, grid, rand_res);
            A.learn(lrnCount);
            goalCount = A.rate(rateCount);
            totalGoalCount += goalCount;
            img.agtCount = i;
            img.goalCount = goalCount;
            img.lrnCount = lrnCount;
            A.writeImg(img);
        }
        cout << ii << endl;
        //output << lrnCount;
        output << agent.p;
        output << ",";
        output << (double)totalGoalCount / agtCount;
        output << endl;
        ii++;
        //agent.p += 0.1;
        //lrnCount += 1000;
        //agent.t +=10;
    }
    output.close();
    return 0;
}