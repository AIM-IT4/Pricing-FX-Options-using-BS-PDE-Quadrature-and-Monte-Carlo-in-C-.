#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

// --------------------------------------------------------------------
// Utility: Standard Normal CDF
// N(x) = 0.5 * erfc(-x/sqrt(2))
// --------------------------------------------------------------------
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// --------------------------------------------------------------------
// 1. Black–Scholes Analytical Price for an FX Call Option
// C = S0 * exp(-r_f*T) * N(d1) - K * exp(-r_d*T) * N(d2)
// d1 = [ln(S0/K) + (r_d - r_f + 0.5*sigma^2)T] / (sigma*sqrt(T))
// d2 = d1 - sigma*sqrt(T)
// --------------------------------------------------------------------
double bsPrice(double S0, double K, double T, double sigma, double r_d, double r_f) {
    double sigmaSqrtT = sigma * sqrt(T);
    double d1 = (log(S0 / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    double d2 = d1 - sigmaSqrtT;
    return S0 * exp(-r_f * T) * norm_cdf(d1) - K * exp(-r_d * T) * norm_cdf(d2);
}

// --------------------------------------------------------------------
// 2. Monte Carlo Pricing (Terminal Value Only)
// For each path, simulate S_T under:
// S_T = S0 * exp[(r_d - r_f - 0.5*sigma^2)*T + sigma*sqrt(T)*Z]
// Compute payoff = max(S_T - K, 0) and discount at exp(-r_d*T)
// --------------------------------------------------------------------
double monteCarloPrice(double S0, double K, double T, double sigma, double r_d, double r_f, int nPaths) {
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> norm(0.0, 1.0);
    double sumPayoffs = 0.0;
    for (int i = 0; i < nPaths; ++i) {
        double z = norm(rng);
        double S_T = S0 * exp((r_d - r_f - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * z);
        double payoff = std::max(S_T - K, 0.0);
        sumPayoffs += payoff;
    }
    return exp(-r_d * T) * (sumPayoffs / nPaths);
}

// --------------------------------------------------------------------
// 2b. Simulate and print a few full sample paths (for illustration)
// Here we simulate the full trajectory using a discrete time grid.
// --------------------------------------------------------------------
void simulatePaths(int nPaths, int nSteps, double S0, double T, double sigma, double r_d, double r_f) {
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> norm(0.0, 1.0);
    double dt = T / nSteps;
    for (int p = 0; p < nPaths; ++p) {
        vector<double> path(nSteps + 1);
        path[0] = S0;
        for (int i = 1; i <= nSteps; ++i) {
            double z = norm(rng);
            path[i] = path[i - 1] * exp((r_d - r_f - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * z);
        }
        cout << "Path " << p + 1 << ": ";
        for (auto s : path)
            cout << s << " ";
        cout << "\n";
    }
}

// --------------------------------------------------------------------
// 3. PDE Pricing Using Crank–Nicolson Scheme
//
// We solve the PDE:
//   V_t + (r_d - r_f)S V_S + 0.5*sigma^2 S^2 V_SS - r_d V = 0
// with terminal condition: V(S, T) = max(S - K, 0)
// and boundary conditions: V(0,t)=0 and V(S_max,t)=S_max - K*exp(-r_d*(T-t))
//
// We discretize S ∈ [0, S_max] into M steps (dS = S_max/M)
// and time into N steps (dt = T/N). For interior nodes i = 1,...,M-1,
// the Crank–Nicolson scheme gives a tridiagonal system solved at each time step.
// --------------------------------------------------------------------
double pdePriceCN(double S0, double K, double T, double sigma, double r_d, double r_f, int M, int N) {
    // Define spatial grid: S ∈ [0, S_max]
    double S_max = 3.0 * std::max(S0, K);
    double dS = S_max / M;
    double dt = T / N;
    
    // Grid vectors for asset prices and option values
    vector<double> S(M + 1), V(M + 1);
    for (int i = 0; i <= M; ++i) {
        S[i] = i * dS;
        // Terminal payoff: European Call = max(S - K, 0)
        V[i] = std::max(S[i] - K, 0.0);
    }
    
    // Precompute constant coefficients for interior grid points (i = 1, ... , M-1)
    // We index these coefficients from 0 to M-2 (i.e., a[i] corresponds to grid i+1)
    vector<double> a(M - 1), b(M - 1), c(M - 1);
    for (int i = 1; i < M; ++i) {
        // Note: S_i = i*dS, but we write coefficients in terms of i.
        double A = r_d - r_f;
        double sigma2 = sigma * sigma;
        // Coefficients for the implicit part (using Crank–Nicolson discretization)
        a[i - 1] = -0.25 * dt * (sigma2 * i * i - A * i);
        b[i - 1] = 1.0 + 0.5 * dt * (sigma2 * i * i + r_d);
        c[i - 1] = -0.25 * dt * (sigma2 * i * i + A * i);
    }
    
    // Temporary vectors for Thomas algorithm (tridiagonal solver)
    vector<double> c_star(M - 1, 0.0), d_star(M - 1, 0.0);
    
    // Backward time-stepping: from n = N-1 down to 0
    for (int n = N - 1; n >= 0; --n) {
        double t = n * dt;
        // Boundary conditions at time t:
        double V0 = 0.0;  // At S = 0
        double VM = S_max - K * exp(-r_d * (T - t));  // At S = S_max
        
        // Build the right-hand side vector d (for interior nodes i = 1, ..., M-1)
        vector<double> d(M - 1, 0.0);
        for (int i = 1; i < M; ++i) {
            double A = r_d - r_f;
            double sigma2 = sigma * sigma;
            double alpha = 0.25 * dt * (sigma2 * i * i - A * i);
            double beta  = 1.0 - 0.5 * dt * (sigma2 * i * i + r_d);
            double gamma = 0.25 * dt * (sigma2 * i * i + A * i);
            d[i - 1] = alpha * V[i - 1] + beta * V[i] + gamma * V[i + 1];
        }
        // Adjust the first and last equations for boundary values
        d[0]    -= a[0] * V0;
        d[M - 2] -= c[M - 2] * VM;
        
        // Thomas algorithm: forward sweep
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];
        for (int i = 1; i < M - 1; ++i) {
            double m = b[i] - a[i] * c_star[i - 1];
            c_star[i] = c[i] / m;
            d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
        }
        // Back substitution
        vector<double> V_inner(M - 1, 0.0);
        V_inner[M - 2] = d_star[M - 2];
        for (int i = M - 3; i >= 0; --i)
            V_inner[i] = d_star[i] - c_star[i] * V_inner[i + 1];
        
        // Update the solution vector V (for interior nodes)
        for (int i = 1; i < M; ++i)
            V[i] = V_inner[i - 1];
        V[0] = V0;
        V[M] = VM;
    }
    
    // Interpolate to get the price at S0 (if S0 is not exactly on the grid)
    int idx = std::min(static_cast<int>(S0 / dS), M - 1);
    double weight = (S0 - S[idx]) / dS;
    double price = V[idx] * (1.0 - weight) + V[idx + 1] * weight;
    return price;
}

// --------------------------------------------------------------------
// 4. Quadrature Pricing Using Simpson's Rule
// Price = exp(-r_d*T) * ∫_K^(S_max) (S - K) f(S) dS,
// where f(S) is the lognormal density under risk–neutral measure.
// --------------------------------------------------------------------
double quadraturePrice(double S0, double K, double T, double sigma, double r_d, double r_f, int nSteps) {
    // Choose an upper limit capturing most of the density
    double S_max = S0 * exp((r_d - r_f) * T + 6.0 * sigma * sqrt(T));
    if (nSteps % 2 == 1) nSteps++; // Simpson's rule requires even number of subintervals
    double h = (S_max - K) / nSteps;
    
    auto density = [=](double S) -> double {
        double mu = (r_d - r_f - 0.5 * sigma * sigma) * T;
        double var = sigma * sigma * T;
        double logTerm = log(S / S0);
        return 1.0 / (S * sqrt(2.0 * M_PI * var)) * exp(-pow(logTerm - mu, 2) / (2.0 * var));
    };
    
    auto integrand = [=](double S) -> double {
        return (S - K) * density(S);
    };
    
    double sum = integrand(K) + integrand(S_max);
    for (int i = 1; i < nSteps; ++i) {
        double x = K + i * h;
        if (i % 2 == 0)
            sum += 2.0 * integrand(x);
        else
            sum += 4.0 * integrand(x);
    }
    double integral = (h / 3.0) * sum;
    return exp(-r_d * T) * integral;
}

// --------------------------------------------------------------------
// Main Function
// --------------------------------------------------------------------
int main() {
    // Option and model parameters
    double S0    = 1.20;    // Spot FX rate (e.g., EUR/USD)
    double K     = 1.25;    // Strike rate
    double T     = 1.0;     // Time to maturity in years
    double sigma = 0.15;    // Volatility
    double r_d   = 0.05;    // Domestic risk-free rate
    double r_f   = 0.02;    // Foreign risk-free rate

    cout << fixed << setprecision(7);
    cout << "Pricing FX Options using various methods:\n";
    
    // 1. Black–Scholes Analytical Price
    double priceBS = bsPrice(S0, K, T, sigma, r_d, r_f);
    cout << "Black-Scholes Price: " << priceBS << "\n";
    
    // 2. Monte Carlo Simulation (Terminal Price)
    int nPaths = 1000000;
    auto startMC = chrono::high_resolution_clock::now();
    double priceMC = monteCarloPrice(S0, K, T, sigma, r_d, r_f, nPaths);
    auto endMC = chrono::high_resolution_clock::now();
    chrono::duration<double> durMC = endMC - startMC;
    cout << "Monte Carlo Price (" << nPaths << " paths): " << priceMC 
         << " [Time: " << durMC.count() << " s]\n";
    
    // 2b. Simulate and print a few full sample paths (e.g., 5 paths with 50 time steps)
    int samplePaths = 5;
    int nSteps      = 50;
    cout << "\nSimulated Full Paths (each with " << nSteps + 1 << " time steps):\n";
    auto startPaths = chrono::high_resolution_clock::now();
    simulatePaths(samplePaths, nSteps, S0, T, sigma, r_d, r_f);
    auto endPaths = chrono::high_resolution_clock::now();
    chrono::duration<double> durPaths = endPaths - startPaths;
    cout << "Sample Paths Generation Time: " << durPaths.count() << " s\n";
    
    // 3. PDE Pricing using Crank-Nicolson
    int M = 200;  // Number of spatial steps
    int N = 1000; // Number of time steps
    auto startPDE = chrono::high_resolution_clock::now();
    double pricePDE = pdePriceCN(S0, K, T, sigma, r_d, r_f, M, N);
    auto endPDE = chrono::high_resolution_clock::now();
    chrono::duration<double> durPDE = endPDE - startPDE;
    cout << "PDE Price (Crank-Nicolson): " << pricePDE 
         << " [Time: " << durPDE.count() << " s]\n";
    
    // 4. Quadrature Price using Simpson's Rule
    int nQuadSteps = 1000;
    double priceQuad = quadraturePrice(S0, K, T, sigma, r_d, r_f, nQuadSteps);
    cout << "Quadrature Price (Simpson's Rule): " << priceQuad << "\n";
    
    return 0;
}
