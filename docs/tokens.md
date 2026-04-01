# Tokenization

## Source data

Expected input format: 1-minute OHLCV bars.

## Features

Aggregate 1-minute bars into non-overlapping $m$-minute bars. For each bar at index $i$, the following features can be computed:

**Z — Volatility-normalized return**

Let $r_t = \ln(c_t / c_{t-1})$ be the 1-minute log return. Realized volatility over a trailing window of $W_\sigma$ 1-minute bars:

$$\sigma_t = \sqrt{m} \cdot \operatorname{std}(r_{t-W_\sigma+1}, \ldots, r_t)$$

The $m$-minute log return is $R_i = \ln(c_{mi+m} / c_{mi})$. The z-score is:

$$z_i = \frac{R_i}{\sigma_{mi+m}}$$

**V — Log volume change**

Let $V_i = \sum_{k=1}^{m} \text{quote\_vol}_{mi+k}$ be the $m$-minute quote volume. The log volume change is:

$$\Delta v_i = \ln\!\left(\frac{V_i}{V_{i-1}}\right)$$

**ToD — Time of day**

The UTC day is divided into $K_\text{ToD}$ equal slots of width $24 / K_\text{ToD}$ hours:

$$\text{tod}_i = \left\lfloor \frac{\text{hour}(t_i)}{24 / K_\text{ToD}} \right\rfloor \in \{0, \ldots, K_\text{ToD}-1\}$$

**DoW — Day of week**

$$\text{dow}_i \in \{0, \ldots, 6\} \quad (0 = \text{Monday})$$

## Binning

Continuous features (Z, V) are discretized using $K$ uniform-quantile bins fitted on the training split only:

$$e_j = Q\!\left(\frac{j}{K}\right), \quad j = 1, \ldots, K-1$$

The bin index of value $x$ is $b = \#\{j : e_j \leq x\} \in \{0, \ldots, K-1\}$. Bin edges are frozen after fitting. Time features (ToD, DoW) are already discrete integers and require no fitting.

## Token encoding

Each feature is assigned a **disjoint range** of tokens from a reserved vocabulary pool of 73 tokens ($\tau_0, \ldots, \tau_{72}$). Feature ranges are allocated left-to-right; a value in bin $b$ of a feature starting at offset $s$ maps to token $\tau_{s+b}$. Because ranges are disjoint, every token is unambiguous regardless of its position in the sequence.

The total number of bins across all features must not exceed 73.

## Temporal delimiters

Rather than using a generic bar-boundary marker, **ToD and DoW tokens serve as delimiters** that simultaneously separate bars and inform the model of the current time. This teaches the model periodic structure directly through the token sequence.

**ToD as bar delimiter.** The ToD token for bar $i$ is emitted immediately before bar $i$'s feature tokens. The model therefore knows the time-of-day slot before it generates or evaluates any feature of that bar. Recurrent intraday patterns (e.g., the behaviour of volume and volatility at the Asian open versus the New York close) become learnable as context-dependent transitions between ToD tokens.

**DoW at day boundaries.** A DoW token is emitted at the start of the sequence and again whenever the day changes (midnight crossing). The model can thus condition on the weekday when generating the first bars of a new day. Weekly seasonality — such as thinner liquidity on Fridays or gap openings on Mondays — is encoded structurally rather than left implicit in positional embeddings.

A sequence spanning two calendar days has the form:

$$\underbrace{D_d}_{\text{day } d}\; \underbrace{T_{t_0}\; f^{(1)}_0 \cdots f^{(n)}_0}_{\text{bar 0}}\; \underbrace{T_{t_1}\; f^{(1)}_1 \cdots f^{(n)}_1}_{\text{bar 1}}\; \cdots\; \underbrace{D_{d+1}}_{\text{midnight}}\; T_{t_k}\; f^{(1)}_k \cdots$$

where $D_d$ is the DoW token for day $d$, $T_{t_i}$ is the ToD token for bar $i$, and $f^{(1)}_i, \ldots, f^{(n)}_i$ are the remaining per-bar feature tokens (e.g., Z, V, TB) in their configured order.
