# Introduction

Below is a complete high‐level C++ example that implements four different methods to price a European FX call option. In FX options, the risk–neutral dynamics of the spot rate $S$ are given by

$$
dS = (r_d - r_f) S\,dt + \sigma S\,dW,
$$

so that under the risk–neutral measure the terminal distribution is lognormal with

$$
\ln\frac{S_T}{S_0} \sim \mathcal{N}\Bigl(\Bigl(r_d - r_f -\frac{1}{2}\sigma^2\Bigr)T,\,\sigma^2T\Bigr).
$$

The well–known analytical price is then given by the adapted Black–Scholes formula

$$
C = S_0 e^{-r_f T} N(d_1) - K e^{-r_dT} N(d_2),
$$

with

$$
d_{1,2} = \frac{\ln\frac{S_0}{K} + \left(r_d - r_f \pm \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}},
$$

and where $N(\cdot)$ is the cumulative normal distribution.

The code below provides:

- **Analytical Price (BS formula)**
- **Monte Carlo simulation:** simulating terminal $S_T$ under risk–neutral drift and averaging the discounted payoff.
- **PDE solution (Explicit Finite Difference):** solving the backward Black–Scholes PDE

  $$
  \frac{\partial V}{\partial t} + (r_d - r_f) S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - r_d V = 0,
  $$

  with final condition $V(S,T) = \max(S-K, 0)$ and appropriate boundary conditions.
- **Quadrature method:** using Simpson’s rule to integrate the discounted payoff against the lognormal density.

## Explanation & Notes

**Black–Scholes:**

We compute $d_1$ and $d_2$ using logarithms and the risk–neutral drift $(r_d - r_f)$ (note the FX adjustment). The option value is then computed using discounted cumulative normal values.

**Monte Carlo:**

We generate $n$ independent paths for $S_T$ using the risk–neutral dynamics. Each path’s payoff is $\max(S_T - K, 0)$ and then we discount by $\exp(-r_dT)$. For production, consider parallelizing the loop.

**PDE (Finite Difference):**

We use an explicit Euler (forward in time) scheme (stepping backwards from maturity) with central difference approximations for the first and second derivatives.

*Note:* This method is conditionally stable; choose $N$ and $M$ so that the stability (CFL) condition is met.

**Quadrature (Simpson's Rule):**

We compute the integral

$$
\int_K^{S_{max}} (S-K) f(S) \, dS,
$$

where $f(S)$ is the lognormal density under the risk–neutral measure. The upper limit $S_{max}$ is chosen to capture most of the density (6 standard deviations).

Compile and run the code with an optimizing compiler flag to achieve high performance. This example is self-contained and demonstrates both the mathematical derivation and the coding implementation for each method. Compile with high optimization (for example, using `-O2` or `-O3`).

Below is the complete code:
