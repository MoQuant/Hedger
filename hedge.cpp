#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>
#include <curl/curl.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cmath>

using namespace boost::property_tree;

std::string address(std::string ticker){
    std::string key = "";
    std::string url = "https://financialmodelingprep.com/api/v3/historical-price-full/" + ticker + "?from=2021-01-21&apikey=" + key;
    return url;
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s)
{
    size_t newLength = size * nmemb;
    try
    {
        s->append((char*)contents, newLength);
    }
    catch (std::bad_alloc& e)
    {
        // Handle memory problem
        return 0;
    }
    return newLength;
}

// Function to perform GET request
std::string Request(std::string ticker)
{
    std::string url = address(ticker);
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    return readBuffer;
}

void TYPOHOON(std::string response, std::string ticker, std::map<std::string, std::vector<double>> & data){
    std::stringstream ss(response);
    ptree X;
    read_json(ss, X);

    for(ptree::const_iterator it = X.begin(); it != X.end(); ++it){
        if(it->first == "historical"){
            for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
                for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                    if(kt->first == "adjClose"){
                        data[ticker].push_back(kt->second.get_value<double>());
                    }
                }
            }
        }
    }

}

std::vector<std::vector<double>> MMULT(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    std::vector<std::vector<double>> result;
    std::vector<double> temp;
    for(int i = 0; i < x.size(); ++i){
        temp.clear();
        for(int j = 0; j < y[0].size(); ++j){
            double sum_each = 0.0;
            for(int k = 0; k < x[0].size(); ++k){
                sum_each += x[i][k]*y[k][j];
            }
            temp.push_back(sum_each);
        }
        result.push_back(temp);
    }
    return result;
}

std::vector<std::vector<double>> TRANSPOSE(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> result;
    std::vector<double> temp;
    for(int i = 0; i < x[0].size(); ++i){
        temp.clear();
        for(int j = 0; j < x.size(); ++j){
            temp.push_back(x[j][i]);
        }
        result.push_back(temp);
    }
    return result;
}

std::vector<std::vector<double>> FACTOR(double a, std::vector<std::vector<double>> x){
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            x[i][j] *= a;
        }
    }
    return x;
}

void PRINTX(std::vector<std::vector<double>> x){
    for(auto & i : x){
        for(auto & j : i){
            std::cout << j << "\t";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> INVERSE(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> I;
    std::vector<double> temp;
    int n = x.size();
    for(int i = 0; i < n; ++i){
        temp.clear();
        for(int j = 0; j < n; ++j){
            if(i == j){
                temp.push_back(1.0);
            } else {
                temp.push_back(0.0);
            }
        }
        I.push_back(temp);
    }

    for(int i = 1; i < n; ++i){
        for(int j = 0; j < i; ++j){
            double A = x[i][j];
            double B = x[j][j];
            for(int k = 0; k < n; ++k){
                x[i][k] -= (A/B)*x[j][k];
                I[i][k] -= (A/B)*I[j][k];
            }
        }
    }

    for(int i = 1; i < n; ++i){
        for(int j = 0; j < i; ++j){
            double A = x[j][i];
            double B = x[i][i];
            for(int k = 0; k < n; ++k){
                x[j][k] -= (A/B)*x[i][k];
                I[j][k] -= (A/B)*I[i][k];
            }
        }
    }

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            I[i][j] /= x[i][i];
        }
    }

    return I;
}

std::vector<std::vector<double>> ADDMATRIX(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b){
    for(int i = 0; i < a.size(); ++i){
        for(int j = 0; j < a[i].size(); ++j){
            a[i][j] += b[i][j];
        }
    }
    return a;
}

std::vector<std::vector<double>> SUBMATRIX(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b){
    for(int i = 0; i < a.size(); ++i){
        for(int j = 0; j < a[i].size(); ++j){
            a[i][j] -= b[i][j];
        }
    }
    return a;
}

std::vector<std::vector<double>> RateOfReturn(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> y;
    std::vector<double> temp;
    for(int i = 1; i < x.size(); ++i){
        temp.clear();
        for(int j = 0; j < x[0].size(); ++j){
            temp.push_back(x[i][j]/x[i-1][j] - 1.0);
        }
        y.push_back(temp);
    }
    return y;
}

std::vector<double> MVPortfolio(std::vector<std::vector<double>> X){
    std::vector<double> weights, bottom;
    int m = X.size(), n = X[0].size();
    std::vector<std::vector<double>> ones, mu, cov, B, W;
    for(int i = 0; i < m; ++i){
        ones.push_back({1.0});
    }
    mu = FACTOR(1.0/(double) m, MMULT(TRANSPOSE(ones), X));
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            X[i][j] -= mu[0][j];
        }
    }

    cov = FACTOR(1.0/((double) m - 1), MMULT(TRANSPOSE(X), X));
    
    for(int i = 0; i < n; ++i){
        cov[i].push_back(1.0);
        bottom.push_back(1.0);
    }
    bottom.push_back(0.0);
    cov.push_back(bottom);

    for(int i = 0; i < n; ++i){
        B.push_back({0.0});
    }
    B.push_back({1.0});

    W = MMULT(INVERSE(cov), B);
    
    for(int i = 0; i < n; ++i){
        weights.push_back(W[i][0]);
    }
    
    return weights;
}

std::vector<double> MXPortfolio(std::vector<std::vector<double>> X){
    std::vector<double> weights, bottom;
    int m = X.size(), n = X[0].size();
    std::vector<std::vector<double>> ones, mu, cov, B, W;
    for(int i = 0; i < m; ++i){
        ones.push_back({1.0});
    }
    mu = FACTOR(1.0/(double) m, MMULT(TRANSPOSE(ones), X));
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            X[i][j] -= mu[0][j];
        }
    }

    cov = FACTOR(1.0/((double) m - 1), MMULT(TRANSPOSE(X), X));
    
    ones.clear();
    for(int i = 0; i < n; ++i){
        ones.push_back({1.0});
    }
    cov = INVERSE(cov);
    cov = MMULT(cov, TRANSPOSE(mu));
    ones = MMULT(TRANSPOSE(ones), cov);

    for(int i = 0; i < n; ++i){
        cov[i][0] /= ones[0][0];
        weights.push_back(cov[i][0]);
    }
    
    return weights;
}


std::vector<double> ComputePortfolio(std::vector<std::vector<double>> X, int window){
    std::vector<double> portfolio, weights;
    std::vector<std::vector<double>> cutitout;

    for(int i = window; i < X.size(); ++i){
        cutitout = {X.begin() + (i - window), X.begin() + i};
        weights = MVPortfolio(cutitout);
        double total = 0;
        for(int j = 0; j < weights.size(); ++j){
            total += weights[j]*X[i][j];
        }
        if(std::isinf(total) || std::isnan(total)){
            total = 0;
        }
        portfolio.push_back(total);
    }

    return portfolio;
}

std::vector<double> HedgeFund(std::vector<double> x, std::vector<double> y){
    
    auto bm = [](double xi){
        std::vector<std::vector<double>> u = {{1.0}, {xi}};
        return u;
    };

    auto mean = [](std::vector<double> u){
        double total = 0;
        for(auto & s : u){
            total += s;
        }
        return total / (double) u.size();
    };
    
    std::vector<std::vector<double>> B, Bp, Pp, Q, P, Yp, K, DK, deltaB;
    std::vector<double> results;
    
    double R = 0;
    int n = x.size();
    int m = y.size();

    B = {{0.1}, {0.1}};
    Q = {{1.0, 0.0},{0.0,1.0}};
    P = {{1.0, 0.0},{0.0,1.0}};

    for(int i = 0; i < n; ++i){
        Bp = B;
        Pp = ADDMATRIX(Q, P);
        Yp = MMULT(TRANSPOSE(Bp), bm(x[i]));
        if(i > 1){
            R = 0;
            for(int q = 0; q < i; ++q){
                R += pow(y[i] - Yp[0][0], 2);
            }
            R /= ((double) i - 1);
        }
        K = MMULT(Pp, bm(x[i]));
        DK = MMULT(TRANSPOSE(K), bm(x[i]));
        DK[0][0] += R;
        for(int g = 0; g < K.size(); ++g){
            K[g][0] /= DK[0][0];
        }
        B = ADDMATRIX(Bp, FACTOR(y[i] - Yp[0][0], K));
        P = SUBMATRIX(Pp, MMULT(K, MMULT(TRANSPOSE(bm(x[i])), Pp)));
        deltaB = SUBMATRIX(B, Bp);
        Q = MMULT(deltaB, TRANSPOSE(deltaB));
    }

    double rss = 0;
    double avg = 0;
    double mu = mean(x);

    for(int i = 0; i < y.size(); ++i){
        rss += pow(y[i] - (B[0][0] +B[1][0]*x[i]), 2);
        avg += pow(x[i] - mu, 2);
    }
    rss /= (double) n - 2;
    double tstat = B[1][0] / sqrt(rss/avg);

    results.push_back(B[1][0]);
    results.push_back(tstat);

    return results;
}

int main()
{
    std::vector<std::string> stock_ticks = {"AAPL","MSFT","NVDA","TSLA","GME","VXX"};

    std::vector<std::string> hedge_ticks = {"BITO","SMH","COPX","SIVR","AIRR","XMMO","SPMO"};

    std::map<std::string, std::string> hedges = {
        {"BITO","ProShares Bitcoin Strategy ETF"},
        {"SMH","VanEck Semiconductor ETF"},
        {"COPX","Global X Copper Miners ETF"},
        {"SIVR","abrdn Physical Silver Shares ETF"},
        {"AIRR","First Trust RBA American Industrial Renaissance ETF"},
        {"XMMO","Invesco S&P MidCap Momentum ETF"},
        {"SPMO","Invesco S&P 500 Momentum ETF"}
    };

    std::map<std::string, std::vector<double>> df;
    std::vector<std::vector<double>> SClose, HClose, SROR, HROR;

    for(auto & ticker : stock_ticks){
        TYPOHOON(Request(ticker), ticker, std::ref(df));
        std::reverse(df[ticker].begin(), df[ticker].end());
        SClose.push_back(df[ticker]);
    }

    for(auto & ticker : hedge_ticks){
        TYPOHOON(Request(ticker), ticker, std::ref(df));
        std::reverse(df[ticker].begin(), df[ticker].end());
        HClose.push_back(df[ticker]);
    }

    int ON = HClose[0].size();
    for(int i = 0; i < SClose.size(); ++i){
        SClose[i] = {SClose[i].end() - ON, SClose[i].end()};
    }

    SROR = RateOfReturn(TRANSPOSE(SClose));
    HROR = RateOfReturn(TRANSPOSE(HClose));

    HROR = TRANSPOSE(HROR);

    
    
    int window = 50;
    std::vector<double> selected_portfolio = ComputePortfolio(SROR, window);
    
    for(int i = 0; i < HROR.size(); ++i){
        HROR[i] = {HROR[i].begin() + window, HROR[i].end()};
        std::vector<double> the_hedge = HedgeFund(selected_portfolio, HROR[i]);
        std::cout << "Your portfolio can be hedged with " << hedges[hedge_ticks[i]] << " with a hedging ratio of " << the_hedge[0] << " whose T-Stat is " << the_hedge[1] << std::endl;
    }
    
    

    return 0;
}
