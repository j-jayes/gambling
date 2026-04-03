# **Machine Learning in Predictive Football Analytics: Architecting a Quantitative Betting Model for the 2026 FIFA World Cup**

## **Introduction: The Paradigm Shift in Quantitative Sports Forecasting**

The landscape of sports analytics and quantitative forecasting has undergone a profound transformation over the past decade. Traditional statistical methods, which historically relied on rudimentary historical averages, static heuristic evaluations, and basic expert intuition, have been rendered largely obsolete by the advent of advanced machine learning (ML), generative artificial intelligence (GenAI), and high-frequency spatiotemporal data processing.1 As the global sports analytics market accelerates toward a projected valuation of over $22 billion by 2030, and the AI-driven sports betting sector anticipates a staggering compound annual growth rate (CAGR) of 21%—pushing it from $10.8 billion in 2025 to past $60 billion by 2034—algorithmic prediction models have transitioned from experimental curiosities to fundamental necessities for establishing an edge in prediction markets.1

The 2026 FIFA World Cup, jointly hosted by the United States, Canada, and Mexico, presents an unprecedented forecasting challenge that requires the utmost sophistication from predictive models. The tournament introduces an expanded 48-team format, significantly increasing the variance in team quality, tactical styles, and potential match outcomes.2 Furthermore, the sheer geographical vastness of the host nations introduces extreme environmental variables—ranging from the high altitude of Mexico City to the severe heat and humidity of Miami and Monterrey—that directly impact player physiology, match pacing, and overall expected output.3

To accurately forecast match outcomes (traditional 1X2 markets), outright tournament winners, and complex proposition bets (prop bets) such as the frequency of corners, cards, and player-specific milestones, a sophisticated, multi-layered machine learning architecture is required. This report provides an exhaustive, highly technical blueprint for constructing a state-of-the-art predictive model for the 2026 FIFA World Cup. It covers the optimal data acquisition pipelines, complex feature engineering methodologies accounting for physiological and environmental stressors, mathematical architectures for modeling highly stochastic events, and the complete technical integration required to execute algorithmic trades autonomously via the Polymarket decentralized prediction market API.

## **The Foundational Data Ecosystem: Sourcing and Structuring Inputs**

A predictive machine learning model is fundamentally constrained by the quality, granularity, reliability, and latency of its data inputs. For a global tournament like the World Cup, the data ecosystem must aggregate decades of historical match results, underlying performance metrics, physiological data, and real-time situational contexts.6

### **Commercial Data Providers and API Architectures**

To build a robust training dataset and fuel real-time inference engines, integration with enterprise-grade sports data APIs is critical. Several providers dominate the market, each offering distinct advantages depending on the quantitative project's capitalization and technical scope. The optimal choice balances data depth, latency, and ease of integration into Python-based machine learning pipelines.8

| Data Provider | Core Strengths | Target Audience & Use Case | Limitations |
| :---- | :---- | :---- | :---- |
| **Sportmonks** | Premier football-specific coverage, developer-friendly REST API, extensive live match statistics (xG, shot maps, lineups), 99.99% uptime guarantee. | Independent quantitative analysts, football-centric betting models, and prop bet forecasting. | Narrower breadth across non-football sports; advanced metrics restricted to premium tiers.8 |
| **Sportradar** | Enterprise-grade infrastructure, unparalleled data depth, official league partnerships. | Major sportsbooks, institutional syndicates, media conglomerates. | Extremely high cost structure; restrictive licensing for independent developers.8 |
| **Highlightly** | Combines statistical match data with video highlights and media features. | Media applications, fan engagement platforms, budget-constrained media projects. | Less emphasis on ultra-low latency quantitative feeds.8 |
| **SportsDataIO** | Broad coverage across multiple professional sports, offering both historical and live API feeds. | Multi-sport betting platforms and cross-sport arbitrage systems. | Football data may lack the extreme granularity provided by soccer-specific APIs.8 |

For the specific purpose of building a proprietary betting model for the 2026 World Cup, Sportmonks emerges as the most efficient solution. It delivers structured JSON responses containing real-time scores, match statistics, team and player metadata, and historical World Cup data necessary for modeling complex prop bets like corners and cards.8

### **Dataset Composition, Web Scraping, and Preprocessing**

A holistic dataset for the 2026 World Cup must trace historical international fixtures back through previous World Cup cycles—at minimum capturing the 1994 to 2022 tournaments to establish a baseline for international tournament dynamics—and include the entirety of the 2026 qualification phases.7 Because international teams play significantly fewer matches than domestic clubs, standard API data is often supplemented with custom web scraping.

Quantitative researchers routinely deploy libraries such as Python's Selenium, Beautiful Soup 4, and Pandas to extract supplementary data from sources such as FBref, Wikipedia, and the World Bank.7 This process involves extracting historical match results, cleaning the data, structuring it into usable matrices, and encoding team strengths to generate possible 2026 group-stage fixtures and knockout round simulations.7

The preprocessing pipeline is a rigorous undertaking. It must handle missing values—often utilizing median imputation or k-nearest neighbors imputation for missing player statistics—encode categorical variables such as team formations and managerial tactical profiles, and normalize continuous variables like possession percentages and passing accuracy.16 Ensuring data consistency and mitigating the noise inherent in international football requires meticulous data cleaning, an area where Artificial Intelligence algorithms are increasingly deployed to automatically clean datasets and enforce consistency across disparate sources.17

## **Advanced Feature Engineering: Capturing Tactical and Contextual Realities**

Feature engineering is the critical nexus where deep domain expertise meets machine learning sophistication.18 Raw data (e.g., goals scored, passes completed) is inherently noisy and often misleading. It must be synthesized into predictive features that capture the true underlying strength, tactical tendencies, and contextual state of the competing teams.

### **Core Performance and Tactical Metrics**

Modern predictive models transcend basic goal differentials and rudimentary statistics, relying heavily on advanced underlying metrics to isolate signal from noise:

1. **Expected Goals (xG) and Expected Threat (xT):** Traditional evaluations of team quality relied heavily on actual goals scored, a metric prone to extreme variance and luck. AI has elevated this process by calculating expected goals (xG). Modern machine learning algorithms factor in a much broader range of variables than earlier iterations, including shot distance, defensive positioning, ball trajectory, weather conditions, and the specific pressure applied by opponents.6 These AI-driven statistics provide a highly reliable projection of a team's true offensive output and the likelihood of future goal-scoring events.6  
2. **Elo Ratings and Momentum Indicators:** Dynamic Elo ratings, which mathematically update a team's strength based on match results and the quality of the opponent, serve as a foundational baseline for team strength.11 However, static ratings are insufficient. Advanced models incorporate momentum indicators that trace the current trajectory of a team.18 These indicators use weighted performance metrics that prioritize recent fixtures, squad continuity, and a winning mentality (such as Argentina's status as defending champions) to capture fluctuations in form.18  
3. **Opposition-Adjusted Statistics:** Raw statistics are heavily skewed by the difficulty of the schedule. For instance, an average corner count of 8.0 against low-tier qualification opponents does not translate to a World Cup group stage. Feature sets must include opposition-adjusted metrics, contextualizing a team's attacking and defensive output against the specific quality of the opponents they have faced.18 When modeling specific prop bets like corners, features such as the away team's average corners conceded, the home team's expected goals, head-to-head corner averages, and the home team's total shots have been identified via algorithms like XGBoost as highly significant indicators.15  
4. **Spatiotemporal and Tracking Data:** Metrics derived from GPS and optical tracking—similar to the Next Gen Stats utilized in the NFL Big Data Bowl—map player trajectories, high-intensity running volume, and formation compactness.21 These models process real-time positional data to evaluate tactical compliance and physical exertion, offering deep insights into how a team maintains its defensive shape or exploits offensive space.22

### **Environmental and Geographical Variables for the 2026 World Cup**

The 2026 World Cup presents unprecedented geographical and climatic challenges, spanning 16 host cities across Canada, Mexico, and the United States.3 A predictive model that fails to account for thermal stress and altitude will suffer from severe omitted variable bias, rendering its predictions dangerously inaccurate. Incorporating these variables into the feature space is non-negotiable for finding an edge in the betting markets.

#### **Extreme Heat, Humidity, and Thermal Stress**

Matches scheduled in cities such as Dallas/Arlington, Houston, Miami, and Monterrey are categorized by climatologists and sports scientists as carrying a 'very high' risk for extreme heat stress.4 Traditional ambient temperature metrics are insufficient for modeling player fatigue. Instead, sophisticated models must utilize the Wet Bulb Globe Temperature (WBGT) and the Universal Thermal Climate Index (UTCI).23

These indices account for a combination of ambient temperature, humidity, solar heat load, and wind speed.23 Furthermore, the UTCI can be adjusted for exercise to consider the metabolic heat produced by the players, their movement speed, and clothing barriers.23 For example, in Miami (Hard Rock Stadium), late June conditions routinely feature temperatures approaching 29°C combined with over 70% humidity.5 These conditions severely penalize high-intensity pressing teams (e.g., Scotland or Germany) that are accustomed to temperate maritime climates.5 Fluid loss accelerates, recovery becomes exponentially harder, and late-game physical performance degrades rapidly. ML models must introduce these variables as decay functions applied to a team's baseline expected goals and defensive solidity in the second half of matches.

#### **Altitude and Hypoxia**

The reduction in the partial pressure of oxygen at high altitudes profoundly limits aerobic capacity, slows recovery from high-intensity efforts, and alters ball aerodynamics.25 Estadio Azteca in Mexico City sits at an elevation of over 2,200 meters. Historical analyses of international football demonstrate that altitude offers a significant, quantifiable advantage to acclimatized teams (such as Mexico or Andean nations).26

To quantify this advantage, analysts utilize generalized linear models to extract altitude adjustments. Studies have shown that altitude features can be integrated directly into match performance formulas. For instance, mathematical models have introduced altitude scalars—such as a coefficient of ![][image1] or ![][image2] (depending on the specific elevation tier)—into the baseline goal expectancy equations, explicitly shifting the probability distribution in favor of the altitude-adapted team.28 The model must also recognize that maximal accelerations may not decrease initially due to player pacing strategies, but transient fatigue increases significantly, meaning teams at altitude are highly vulnerable to late-game defensive collapses.26

#### **Travel Fatigue and Circadian Disruption**

The 2026 tournament will expose players to extensive international travel and jet lag.3 The cumulative load of traversing multiple time zones and covering thousands of miles between group stage matches disrupts circadian rhythms and reduces sleep quality, directly impacting technical execution and tactical decision-making.3 Algorithms should calculate the cumulative travel distance and the number of recovery days between matches for each team, inputting these metrics as negative weight features that suppress offensive efficiency ratings.30

### **Feature Selection and Dimensionality Reduction**

To prevent overfitting—a common and fatal pitfall when modeling tournaments with relatively small sample sizes like the World Cup—rigorous feature selection techniques are paramount. Techniques such as the BORUTA algorithm are frequently employed to identify statistically significant features. BORUTA works by comparing the importance of original attributes with randomized "shadow" features.31 By iteratively removing highly correlated features and eliminating those with overlapping significance boundaries, the algorithm ensures the model retains a lean, highly predictive feature set (e.g., isolating definitive metrics like total shots, crosses, expected goals, and opposition tactical setups while discarding noisy, irrelevant data).15

## **Algorithmic Frameworks for Match Prediction**

The fundamental nature of football is highly stochastic. A single low-probability event—such as an early red card, a penalty kick, or a heavily deflected shot—can entirely dictate the outcome of a 90-minute match. Therefore, predictive models cannot merely output a binary classification of a "winner." They must output precise, well-calibrated probability distributions.16

### **Predicting Match Outcomes (The 1X2 Market)**

The foundational market in sports betting is the 1X2 market, representing a Home Win, a Draw, or an Away Win. Modern quantitative architectures address this multi-class classification problem through advanced statistical ensembles.

#### **Bayesian Logistic Regression**

Unlike frequentist logistic regression, which outputs a single, rigid point estimate, Bayesian logistic regression provides a full posterior distribution of probabilities. This framework is highly adaptable and robust to the uncertainties inherent in tournament football.16 It allows analysts to formally incorporate prior knowledge into the model. For instance, the historical tournament pedigree of a nation, or the subjective "Messi factor" characterizing a team's leadership and tournament experience, can be encoded as prior probability distributions.16 As the tournament progresses and new match data is ingested, the model dynamically updates these priors to form a posterior distribution. This explicit quantification of uncertainty is mathematically crucial for identifying value bets in financial markets.16

#### **Gradient Boosting Ensembles (XGBoost, CatBoost, Random Forest)**

Tree-based ensemble methods, particularly Extreme Gradient Boosting (XGBoost) and Random Forests, have consistently demonstrated state-of-the-art performance in sports prediction.7 In backtesting scenarios on historical World Cup datasets, baseline XGBoost models with hyperparameter tuning have achieved test accuracies of approximately 68.75% on out-of-sample tournaments, with some highly optimized models in broader football forecasting reaching classification accuracies above 82%.11

These algorithms excel because they effectively capture complex, non-linear relationships within the data. For example, XGBoost can automatically model the interaction between a team's deep defensive block and an opponent's high transition speed without requiring the analyst to manually specify the interaction term. Furthermore, they are favored for their interpretability; researchers can extract feature importance metrics to understand exactly which variables are driving the predictions.22

However, modeling 1X2 outcomes introduces a significant challenge: class imbalance, specifically regarding the "Draw" classification. Matches between closely matched teams often result in draws, but standard machine learning models frequently suffer from abysmal recall rates for draws (sometimes as low as 5.6%).11 The algorithm tends to default to predicting a decisive winner because the margin of error in football is so narrow.11 To counteract this, quantitative developers must utilize custom loss functions, apply synthetic minority over-sampling techniques (SMOTE), or integrate the classification outputs with Poisson-based count models to better calibrate the probability of a draw.11

#### **Deep Learning (ANNs, CNNs, LSTMs)**

While Deep Learning architectures such as Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), and Long Short-Term Memory (LSTM) networks are utilized for highly specific tasks—such as tracking a player's exact spatial trajectory from video feeds or forecasting a player's goal-scoring totals over a long time horizon—they are generally less favored than ensemble trees for overarching match outcome predictions.22 Deep learning models require massive volumes of data to train effectively and act largely as "black boxes," making it difficult to interpret why a specific probability was assigned. In the context of international football, where the sample size of relevant matches is relatively small, deep learning models are highly susceptible to overfitting.22

## **Stochastic Modeling for Proposition Bets**

While 1X2 markets are the most liquid, proposition bets (prop bets) often present greater inefficiencies and higher expected value for sharp bettors. Predicting the exact number of goals, corners, yellow cards, or individual player milestones demands a shift from classification algorithms to complex count data modeling. These events represent discrete, non-negative integers that naturally follow specific mathematical probability distributions.

### **Predicting Goals: The Poisson Distribution and the Dixon-Coles Adjustment**

The baseline mathematical standard for predicting the number of goals scored by a team is the Poisson regression model.35 The Poisson distribution calculates the probability of a given number of events occurring in a fixed interval of time, assuming these events occur with a known constant mean rate (![][image3]) and independently of the time since the last event.

The probability of a team scoring ![][image4] goals is calculated as:

![][image5]  
In a bivariate Poisson model used for football, the home team's goal expectancy (![][image6]) is defined as a function of their attacking strength, the away team's defensive vulnerability, and a quantified home-field advantage modifier. A double Poisson regression models the two teams independently, allowing the analyst to simulate the probability of every possible scoreline (e.g., the probability of the home team scoring 2 goals multiplied by the probability of the away team scoring 1 goal yields the probability of a 2-1 exact score).33

However, the assumption of strict independence in a pure Poisson model is a known fallacy in football analytics. Goals are not truly independent events. A team leading 2-0 will frequently shift to a conservative tactical posture to protect the lead, altering the underlying scoring rate for the remainder of the match. Furthermore, empirical data shows that standard Poisson models consistently under-predict the frequency of low-scoring draws (0-0, 1-1) and narrow victories (1-0, 0-1).33

To rectify this structural flaw, the **Dixon-Coles model** is applied. Dixon and Coles introduced an adjustment factor that applies a specific modification function, ![][image7], to the standard bivariate Poisson probabilities. This mathematical adjustment explicitly inflates the probability of 0-0, 1-0, 0-1, and 1-1 outcomes while simultaneously deflating the probabilities of higher-scoring, independent permutations.13 Implementing the Dixon-Coles adjustment is a mandatory step for accurately pricing under/over goal markets and exact score prop bets.

### **Managing Overdispersion in Corners and Cards: Negative Binomial Distributions**

While the adjusted Poisson distribution is highly effective for modeling goals, it structurally fails when predicting higher-frequency, highly volatile events such as corner kicks and yellow cards.15 A fundamental assumption of the Poisson distribution is equidispersion—meaning that the mean of the distribution must equal its variance.

In reality, football match data for corners and cards exhibits profound **overdispersion**, where the variance heavily exceeds the mean.34 For example, the number of corners in a match is highly dependent on game state and tactical matchups. A match featuring a team trailing by a goal late in the game, utilizing aggressive wing play and crossing, may result in 15 corners. Conversely, a cautious, centrally congested match between two defensive teams may yield only 3 corners. Similarly, the distribution of yellow cards is dictated by external, highly variable factors: the historical strictness of the assigned referee, the tactical fouling strategy employed by the underdog, and the emotional stakes of a knockout match.40

To handle this massive overdispersion, the **Negative Binomial distribution** is employed as the standard mechanism.34 The Negative Binomial regression model introduces an additional parameter that allows the variance to scale independently of the mean, providing a much tighter, more accurate fit for predicting corner and card distributions.34 Generalized Additive Models (GAMs) are frequently paired with Negative Binomial distributions to interpret the partial effect of specific covariates (like an aggressive referee or a high-possession team) on the final count, allowing for clear statistical inference.34

Furthermore, when modeling events that frequently result in a zero count (e.g., the probability of a red card occurring, or a specific defender scoring a goal), **Zero-Inflated Poisson (ZIP)** or **Zero-Inflated Negative Binomial (ZINB)** models are utilized.34 These dual-component models use a logistic regression element to first predict the probability of a structural zero (the baseline chance the event simply does not occur) and then apply the count distribution only if the event does occur.34

### **Player-Specific Prop Predictions**

The application of machine learning has also supercharged the prediction of individual player prop bets, such as total passing yards, shots on target, or anytime goalscorer markets.41 Models designed for player props ingest historical player tracking data, usage rates within the team's system, and specific defensive matchup metrics.41

The output of a player prop model involves predicting a central statistical value (e.g., a striker is expected to take 2.4 shots) and then plotting a normal or skewed distribution around the model residuals. By computing the area under the curve, the algorithm calculates the exact probability of a player exceeding a sportsbook's defined threshold (e.g., evaluating the true odds that the striker achieves *over* 1.5 shots on target).44

## **Expected Value (+EV) Optimization and Market Inefficiencies**

Generating an accurate mathematical prediction is only the first half of the quantitative betting equation. The second, arguably more critical component, is identifying and systematically exploiting inefficiencies in the market pricing. Professional algorithmic betting entirely disregards the concept of "picking a winner" in a vacuum. Instead, it operates strictly on the financial and mathematical concept of Positive Expected Value (+EV).45

### **The Mathematics of Positive Expected Value (+EV)**

A positive Expected Value wager exists strictly when the proprietary machine learning model's estimated probability for a specific outcome exceeds the implied probability derived from the market's offered odds (or, in the case of prediction markets, the share price).45

For instance, consider a market on Polymarket for "Argentina to beat Mexico." If a "Yes" share is currently trading at a midpoint price of ![][image8], the decentralized market is implying a 52% probability of an Argentina victory.45 If the proprietary XGBoost and Bayesian ensemble—having ingested the historical data, applied altitude adjustments for the match in Monterrey, and evaluated current xG metrics—calculates Argentina's true win probability to be 60% (![][image9]), the model has identified an edge of 8% against the market.45

The Expected Value equation determines the long-term profitability of a standard 1-unit wager:

![][image10]  
Applied to the mechanics of a Polymarket share, which pays out a flat ![][image11] upon successful resolution:

![][image12]  
![][image13]  
![][image14]  
A calculated positive expected value of ![][image15] indicates an 8% expected return on investment over the long term for this specific wager.45 This rigorous mathematical discipline—relying entirely on the delta between the model's posterior distributions and the market's implied probability, rather than subjective human conviction—is the sole mechanism capable of ensuring sustained profitability and overcoming variance.45

### **Generative AI and Commercial AI Betting Assistants**

As prediction markets and sportsbooks mature, they become increasingly efficient. To maintain an edge, quantitative strategists and retail bettors alike are incorporating Generative AI modules into their pipelines.1 Commercial platforms like Pikkit, Outlier.bet, Rithmm, and BettingPros' Sharp AI have brought machine learning directly to the consumer.41

These AI-driven applications sync with a user's betting history, utilize LLMs to parse conversational queries (e.g., "Which team has the best value in Group A?"), and rapidly scan thousands of available prop lines across various books to identify \+EV opportunities.41 For institutional models, GenAI is deployed for high-speed sentiment analysis and unstructured data parsing. These systems scrape global news feeds, press conferences, and localized social media to detect unannounced injuries, managerial unrest, or tactical shifts long before they are reflected in structured API data or market odds, providing a vital latency advantage.1

## **Decentralized Prediction Markets: The Polymarket Infrastructure**

With predictive models generating highly accurate probabilities and EV thresholds defined, the execution phase requires interfacing with live betting markets. Historically, quantitative bettors relied on traditional sportsbooks. However, traditional books extract a heavy margin (the "vig" or overround) and routinely ban or limit successful algorithmic bettors.

In response, Polymarket has emerged as the premier venue for executing algorithmic sports trades. Polymarket is a decentralized prediction markets platform that enables participants to trade shares in the outcomes of future events using cryptocurrency collateral—specifically USDC.e—on the Polygon blockchain.48 Polymarket operates as a decentralized Central Limit Order Book (CLOB), matching peer-to-peer trades off-chain for speed and settling them on-chain. This structure ensures that market prices organically reflect the crowd's probability estimates without prohibitive institutional margins.48 Outcomes are conclusively determined by the UMA Optimistic Oracle, a decentralized mechanism that verifies real-world results (such as the final score of a World Cup match) and triggers the smart contracts to pay out ![][image11] per winning share.48

### **Technical Architecture of the Polymarket API**

As of 2026, the Polymarket API provides a mature, comprehensive technical interface for developers to build automated prediction market applications.48 The architecture is divided into three highly modular REST components and a real-time WebSocket service.48

#### **1\. Gamma API (Market Metadata & Discovery)**

The Gamma API (https://gamma-api.polymarket.com) serves as the read-only discovery engine.48 Because the Polymarket smart contracts interact purely with hexadecimal hashes, the Gamma API is essential for mapping complex on-chain Conditional Token Framework (CTF) data to human-readable metadata, categories, and tags.48

* **Events vs. Markets:** The API structures data hierarchically. An *Event* is a top-level container grouping related outcomes (e.g., "Who will win the 2026 FIFA World Cup?"). A *Market* is the specific, tradable binary outcome within that event (e.g., "Spain (Yes/No)").50 Each market maps to a unique Condition ID and a pair of ERC-1155 Token IDs (one for Yes, one for No).50  
* **Discovery and Filtering:** Algorithms query the Gamma API using human-readable slugs extracted from the URL, or via sports-specific tag IDs.50 For the World Cup, a bot will query GET /sports to fetch the metadata for football, then execute a filtered query such as GET /events?tag\_id=100381 to retrieve a comprehensive list of all active matches, extracting the Token IDs necessary for trading.52

#### **2\. CLOB API (Central Limit Order Book & Trading)**

The CLOB API (https://clob.polymarket.com) manages live price data and order execution. It interacts with an off-chain matching engine, allowing algorithms to submit, modify, and cancel orders instantly without waiting for blockchain block times or paying exorbitant gas fees for every adjustment.48

* **Pricing Data:** To calculate EV, the algorithm continuously fetches the best bid and ask prices via GET /price, full order book depth via GET /book, and the current implied probability via GET /midpoint.48 If the bid-ask spread is wider than ![][image16], the API intelligently surfaces the last traded price instead of the midpoint to prevent distorted probability calculations.47  
* **Order Execution:** Write operations require user authentication. Orders are submitted via POST /order.48 Crucially for sports markets, the API is designed so that outstanding limit orders are automatically cancelled by the protocol the exact moment the match begins, clearing the order book and preventing traders from being exploited by live events before they can manually cancel.50

#### **3\. Data API & WebSocket Feeds**

The Data API (https://data-api.polymarket.com) focuses on account-specific portfolio management. It allows the algorithmic bot to monitor its current holdings (GET /positions), analyze historical trades, and track overall portfolio value.48 Simultaneously, the WebSocket feed (wss://ws-subscriptions-clob.polymarket.com/ws/) provides real-time data streaming for order book changes and trade executions.48 This push-based architecture allows the algorithm to react to sharp line movements instantly, avoiding the latency and rate limits associated with continuous REST polling.

### **Programmatic Integration and Python Deployment**

Integrating the Polymarket API via Python is streamlined by the official open-source @polymarket/py-clob-client SDK.54

To deploy a betting model, the developer must first establish blockchain authentication. The SDK supports two primary wallet architectures:

1. **Externally Owned Accounts (EOAs):** Standard wallets where the user pays their own gas fees (in POL) for on-chain settlements.56  
2. **Proxy Wallets and the Relayer Client:** Advanced setups utilize Polymarket's gasless relayer. By routing orders through a proxy wallet or participating in the Polymarket Builder Program, the platform covers the on-chain gas fees for trade execution and CTF operations, heavily incentivizing high-frequency algorithmic volume.55

A standard automated trading loop involves initializing the client, deriving Layer 2 (L2) API credentials from the wallet's private key, fetching the market midpoints, comparing them against the internal ML model, and submitting signed orders.

**Conceptual Python Implementation Pipeline:**

Python

from py\_clob\_client.client import ClobClient  
from py\_clob\_client.clob\_types import OrderArgs, OrderType  
from py\_clob\_client.order\_builder.constants import BUY

\# 1\. Initialize Client and Authenticate via Proxy Wallet for Gasless Trades  
client \= ClobClient(  
    host="https://clob.polymarket.com",  
    key="PRIVATE\_KEY\_ENV\_VAR",  
    chain\_id=137, \# Polygon Mainnet  
    signature\_type=1, \# 1 designates Proxy/Smart wallets  
    funder="PROXY\_WALLET\_ADDRESS"  
)  
client.set\_api\_creds(client.create\_or\_derive\_api\_creds())

\# 2\. Fetch Implied Probability (Midpoint) for a specific World Cup Market  
target\_token\_id \= "FETCHED\_TOKEN\_ID\_FROM\_GAMMA\_API"  
midpoint\_data \= client.get\_midpoint(target\_token\_id)  
market\_implied\_prob \= float(midpoint\_data.get('mid'))

\# 3\. ML Inference Comparison (The \+EV check)  
\# The proprietary XGBoost model outputs its probability: model\_prob \= 0.60  
\# If the market\_implied\_prob (e.g., 0.52) shows sufficient EV threshold:

\# 4\. Execute the Trade  
order\_args \= OrderArgs(  
    price=0.53, \# Limit price set slightly above midpoint to ensure execution  
    size=100.0, \# Position sizing dictated by Kelly Criterion  
    side=BUY,  
    token\_id=target\_token\_id  
)  
\# The client automatically handles the tickSize ("0.01") and negRisk parameters  
signed\_order \= client.create\_order(order\_args)  
response \= client.post\_order(signed\_order)

Note: This architecture demonstrates the seamless bridge between off-chain algorithmic decision-making and on-chain decentralized execution.48

## **Strategic Deployment: 2026 World Cup Market Dynamics**

As of early 2026, the qualification phases have concluded, and the field of 48 teams has been finalized.2 Consequently, the liquidity on Polymarket for the "2026 FIFA World Cup Winner" market has surged, representing hundreds of millions of dollars in volume.61

The current outright favorites on Polymarket dictate the baseline expectations and the overarching market sentiment. Quantitative models must weigh these implied probabilities against their internal Elo and momentum calculations to seek arbitrage or value opportunities.

| Nation | Polymarket Implied Probability | Share Price | Top 10 FIFA World Ranking (April 2026\) |
| :---- | :---- | :---- | :---- |
| Spain | 16% | $0.16 | 2nd |
| France | 13% \- 14% | $0.13 \- $0.14 | 1st |
| England | 12% | $0.12 | 4th |
| Argentina | 9% | $0.09 | 3rd |
| Brazil | 8% \- 9% | $0.08 \- $0.09 | 6th |
| Portugal | 6% \- 7% | $0.06 \- $0.07 | 5th |
| Germany | 5% \- 6% | $0.05 \- $0.06 | 10th |
| Data derived from Polymarket order books and FIFA rankings as of early 2026\.2 |  |  |  |

The data reveals slight discrepancies between the official FIFA rankings and the decentralized market sentiment. For example, Spain is currently priced as the outright favorite (16%) by the market, despite France holding the number one position in the FIFA world rankings.2 Similarly, Argentina, the defending champions, are priced below England, indicating that the market may be applying a negative weight to the aging profile of the Argentine squad or recognizing the sheer depth of the European teams.2 These exact discrepancies are the hunting ground for the machine learning models. By rigorously analyzing whether Spain's 16% implied probability accurately reflects their underlying xG and defensive metrics in the context of their specific draw and environmental routing, the algorithm identifies the optimal capital allocation.

## **Conclusion**

The construction of a quantitative machine learning architecture capable of successfully predicting outcomes and complex proposition bets for the 2026 FIFA World Cup is a rigorous, multidisciplinary endeavor. It requires moving far beyond superficial statistics to embrace advanced expected metrics, deep probabilistic mathematics, and highly specific feature engineering.

The unique format of the 2026 World Cup demands that developers construct features capable of quantifying severe environmental stressors—such as the WBGT heat index in Miami and hypoxia at Estadio Azteca—into actionable model weights. Furthermore, the deployment of Bayesian logistic regression and XGBoost for 1X2 markets, coupled with Zero-Inflated Negative Binomial models for overdispersed prop bets like corners and cards, provides the mathematical rigor necessary to outmaneuver the crowd.

By combining these cutting-edge predictive inference engines with the decentralized, programmatic, and low-latency trading infrastructure of the Polymarket API, quantitative analysts can systematically identify positive expected value. This architecture strips human bias entirely from the equation, replacing it with a continuous loop of data ingestion, EV optimization, and automated smart-contract execution. As the 2026 tournament unfolds, the models that can most efficiently ingest real-time tactical shifts, correctly scale for the unprecedented variance of the 48-team format, and seamlessly execute trades in the micro-seconds before market correction will dictate the bleeding edge of global sports analytics.

#### **Works cited**

1. AI Sports Predictions for 2026: Why Traditional Methods Are Now Obsolete \- WSC Sports, accessed on April 3, 2026, [https://wsc-sports.com/blog/industry-insights/ai-sports-predictions-for-2026-why-traditional-methods-are-now-obsolete/](https://wsc-sports.com/blog/industry-insights/ai-sports-predictions-for-2026-why-traditional-methods-are-now-obsolete/)  
2. Who will win the 2026 World Cup? Spain leads early Polymarket predictions as all 48 teams are set, accessed on April 3, 2026, [https://www.fox5dc.com/news/who-will-win-2026-world-cup-spain-leads-early-polymarket-predictions-all-48-teams-set](https://www.fox5dc.com/news/who-will-win-2026-world-cup-spain-leads-early-polymarket-predictions-all-48-teams-set)  
3. New research provides blueprint to protect player health and performance at the 2026 Men's FIFA World Cup \- Loughborough University, accessed on April 3, 2026, [https://www.lboro.ac.uk/news-events/news/2026/march/protecting-player-health/](https://www.lboro.ac.uk/news-events/news/2026/march/protecting-player-health/)  
4. Which 2026 World Cup Host Cities Face the Highest Heat Risk? \- Kestrel, accessed on April 3, 2026, [https://kestrelinstruments.com/blog/which-2026-world-cup-host-cities-face-the-highest-heat-risk](https://kestrelinstruments.com/blog/which-2026-world-cup-host-cities-face-the-highest-heat-risk)  
5. World Cup 2026 Climate Conditions: Temperature, Altitude and Humidity at Every Stadium, accessed on April 3, 2026, [https://talesofthestands.com/2026/03/15/world-cup-2026-climate-conditions-temperature-altitude-and-humidity-across-the-16-stadiums/](https://talesofthestands.com/2026/03/15/world-cup-2026-climate-conditions-temperature-altitude-and-humidity-across-the-16-stadiums/)  
6. Why machine learning football predictions are the new standard ..., accessed on April 3, 2026, [https://www.sportmonks.com/blogs/machine-learning-football-prediction/](https://www.sportmonks.com/blogs/machine-learning-football-prediction/)  
7. EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026 \- GitHub, accessed on April 3, 2026, [https://github.com/EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026](https://github.com/EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026)  
8. Best Sport APIs in 2026 \- Highlightly, accessed on April 3, 2026, [https://highlightly.net/blogs/best-sport-apis-in-2026](https://highlightly.net/blogs/best-sport-apis-in-2026)  
9. The best World Cup 2026 API. All World Cup data for your app. \- Sportmonks, accessed on April 3, 2026, [https://www.sportmonks.com/football-api/world-cup-api/world-cup-2026/](https://www.sportmonks.com/football-api/world-cup-api/world-cup-2026/)  
10. World Cup API: Score the best World Cup Football Data \- Sportmonks, accessed on April 3, 2026, [https://www.sportmonks.com/football-api/world-cup-api/](https://www.sportmonks.com/football-api/world-cup-api/)  
11. Built my first ML model to predict World Cup matches \- 68.75% accuracy. Is this actually good? : r/learnmachinelearning \- Reddit, accessed on April 3, 2026, [https://www.reddit.com/r/learnmachinelearning/comments/1pcmn2g/built\_my\_first\_ml\_model\_to\_predict\_world\_cup/](https://www.reddit.com/r/learnmachinelearning/comments/1pcmn2g/built_my_first_ml_model_to_predict_world_cup/)  
12. 2026 FIFA World Cup qualification \- Wikipedia, accessed on April 3, 2026, [https://en.wikipedia.org/wiki/2026\_FIFA\_World\_Cup\_qualification](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification)  
13. What algorithm should I use for my football game prediction bot? : r/algobetting \- Reddit, accessed on April 3, 2026, [https://www.reddit.com/r/algobetting/comments/1qvmfxq/what\_algorithm\_should\_i\_use\_for\_my\_football\_game/](https://www.reddit.com/r/algobetting/comments/1qvmfxq/what_algorithm_should_i_use_for_my_football_game/)  
14. Predicting Fan Attendance at Mega Sports Events—A Machine Learning Approach: A Case Study of the FIFA World Cup Qatar 2022 \- MDPI, accessed on April 3, 2026, [https://www.mdpi.com/2227-7390/12/6/926](https://www.mdpi.com/2227-7390/12/6/926)  
15. danielsaed/futbol\_corners\_forecast: Machine learning XGBoost regresison model that predicts corners from football matches \- GitHub, accessed on April 3, 2026, [https://github.com/danielsaed/futbol\_corners\_forecast](https://github.com/danielsaed/futbol_corners_forecast)  
16. (PDF) A Bayesian approach for predicting match outcomes: FIFA World Cup 2026, accessed on April 3, 2026, [https://www.researchgate.net/publication/389390461\_A\_Bayesian\_approach\_for\_predicting\_match\_outcomes\_FIFA\_World\_Cup\_2026](https://www.researchgate.net/publication/389390461_A_Bayesian_approach_for_predicting_match_outcomes_FIFA_World_Cup_2026)  
17. Football Data Trends 2026: AI, Player Tracking & What's Next \- Sportmonks, accessed on April 3, 2026, [https://www.sportmonks.com/blogs/football-data-trends-2026-ai-player-tracking-whats-next/](https://www.sportmonks.com/blogs/football-data-trends-2026-ai-player-tracking-whats-next/)  
18. Revolutionizing Football Predictions: How Machine Learning is Changing the Game | by fred Blum | Medium, accessed on April 3, 2026, [https://medium.com/@davidblum\_6849/revolutionizing-football-predictions-how-machine-learning-is-changing-the-game-226b986babae](https://medium.com/@davidblum_6849/revolutionizing-football-predictions-how-machine-learning-is-changing-the-game-226b986babae)  
19. WC2026 Match Probability Baseline Dataset \- Kaggle, accessed on April 3, 2026, [https://www.kaggle.com/datasets/sarazahran1/wc2026-match-probability-baseline-dataset](https://www.kaggle.com/datasets/sarazahran1/wc2026-match-probability-baseline-dataset)  
20. World Cup 2026 Predictions: Who is Going to Win Based on Football Data? \- Sportmonks, accessed on April 3, 2026, [https://www.sportmonks.com/blogs/world-cup-2026-predictions-who-is-going-to-win-based-on-football-data/](https://www.sportmonks.com/blogs/world-cup-2026-predictions-who-is-going-to-win-based-on-football-data/)  
21. Big Data Bowl \- NFL Football Operations, accessed on April 3, 2026, [https://operations.nfl.com/gameday/analytics/big-data-bowl/](https://operations.nfl.com/gameday/analytics/big-data-bowl/)  
22. Machine Learning Applied to Professional Football: Performance ..., accessed on April 3, 2026, [https://www.mdpi.com/2504-4990/7/3/85](https://www.mdpi.com/2504-4990/7/3/85)  
23. Most sites in North America that will host the 2026 World Cup are at high risk of extreme heat, accessed on April 3, 2026, [https://sciencemediacentre.es/en/most-sites-north-america-will-host-2026-world-cup-are-high-risk-extreme-heat](https://sciencemediacentre.es/en/most-sites-north-america-will-host-2026-world-cup-are-high-risk-extreme-heat)  
24. Forecasting thermal stress for sports tourists at the 2026 FIFA World Cup \- PMC, accessed on April 3, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11599364/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11599364/)  
25. Impact of Altitude and Heat on Football Performance \- Gatorade Sports Science Institute, accessed on April 3, 2026, [https://www.gssiweb.org/sports-science-exchange/article/sse-131-impact-of-altitude-and-heat-on-football-performance](https://www.gssiweb.org/sports-science-exchange/article/sse-131-impact-of-altitude-and-heat-on-football-performance)  
26. Soccer activity profile of altitude versus sea-level natives during acclimatisation to 3600 m (ISA3600) \- PMC, accessed on April 3, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3903145/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3903145/)  
27. Moderate Altitude Affects High Intensity Running Performance in a Collegiate Women's Soccer Game, accessed on April 3, 2026, [https://digitalcommons.lmu.edu/cgi/viewcontent.cgi?article=1029\&context=hhsc\_fac](https://digitalcommons.lmu.edu/cgi/viewcontent.cgi?article=1029&context=hhsc_fac)  
28. Effect of altitude on physiological performance: a statistical analysis using results of international football games \- PMC, accessed on April 3, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2151172/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2151172/)  
29. Running Altitude Adjustment Calculator \- Final Surge, accessed on April 3, 2026, [https://www.finalsurge.com/altitude-conversion-calculator](https://www.finalsurge.com/altitude-conversion-calculator)  
30. FIFA World Cup 2026: Performance under Pressure – Scientific Guidelines for Success, accessed on April 3, 2026, [https://www.researchgate.net/publication/397433427\_FIFA\_World\_Cup\_2026\_Performance\_under\_Pressure\_-\_Scientific\_Guidelines\_for\_Success](https://www.researchgate.net/publication/397433427_FIFA_World_Cup_2026_Performance_under_Pressure_-_Scientific_Guidelines_for_Success)  
31. Predicting Football Match Outcomes With Machine Learning Approaches \- Semantic Scholar, accessed on April 3, 2026, [https://pdfs.semanticscholar.org/73df/5c022ec8029e264b34121d6237c952921615.pdf](https://pdfs.semanticscholar.org/73df/5c022ec8029e264b34121d6237c952921615.pdf)  
32. Machine Learning Model for NFL Betting (Model 5.0) | by The Factory of Sadness | Medium, accessed on April 3, 2026, [https://medium.com/@bravenewworld21/machine-learning-model-for-nfl-betting-model-5-0-8e916428c330](https://medium.com/@bravenewworld21/machine-learning-model-for-nfl-betting-model-5-0-8e916428c330)  
33. Predicting Football Results Using Python and the Dixon and Coles Model | penaltyblog, accessed on April 3, 2026, [https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/](https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/)  
34. Leveraging Minute-by-Minute Soccer Match Event Data to Adjust Team's Offensive Production for Game Context \- arXiv, accessed on April 3, 2026, [https://arxiv.org/html/2508.04008v1](https://arxiv.org/html/2508.04008v1)  
35. (PDF) Using Poisson model for goal prediction in European football \- ResearchGate, accessed on April 3, 2026, [https://www.researchgate.net/publication/342442670\_Using\_Poisson\_model\_for\_goal\_prediction\_in\_European\_football](https://www.researchgate.net/publication/342442670_Using_Poisson_model_for_goal_prediction_in_European_football)  
36. Predicting Football Match Results Using a Poisson Regression Model \- MDPI, accessed on April 3, 2026, [https://www.mdpi.com/2076-3417/14/16/7230](https://www.mdpi.com/2076-3417/14/16/7230)  
37. Predicting Football Results With Statistical Modelling \- dashee87.github.io, accessed on April 3, 2026, [https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/)  
38. Amar0302/FootballMatchPredictionPoisson: Predicting football matches using Poisson Distribution based on historical data. \- GitHub, accessed on April 3, 2026, [https://github.com/Amar0302/FootballMatchPredictionPoisson](https://github.com/Amar0302/FootballMatchPredictionPoisson)  
39. FORECASTING FOOTBALL CORNER ODDS \- Lund University Publications, accessed on April 3, 2026, [https://lup.lub.lu.se/student-papers/record/9127007/file/9127013.pdf](https://lup.lub.lu.se/student-papers/record/9127007/file/9127013.pdf)  
40. Predicting Football Results Using Machine Learning Techniques \- Imperial College London, accessed on April 3, 2026, [https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/distinguished-projects/1718-ug-projects/Corentin-Herbinet-Using-Machine-Learning-techniques-to-predict-the-outcome-of-profressional-football-matches.pdf](https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/distinguished-projects/1718-ug-projects/Corentin-Herbinet-Using-Machine-Learning-techniques-to-predict-the-outcome-of-profressional-football-matches.pdf)  
41. Machine Learning Sports Predictions Behind Big Wins \- WSC Sports, accessed on April 3, 2026, [https://wsc-sports.com/blog/industry-insights/machine-learning-sports-predictions-behind-big-wins/](https://wsc-sports.com/blog/industry-insights/machine-learning-sports-predictions-behind-big-wins/)  
42. Lessons From Building a Winning Prop Prediction System : r/algobetting \- Reddit, accessed on April 3, 2026, [https://www.reddit.com/r/algobetting/comments/1hw38vo/lessons\_from\_building\_a\_winning\_prop\_prediction/](https://www.reddit.com/r/algobetting/comments/1hw38vo/lessons_from_building_a_winning_prop_prediction/)  
43. Modeling for Sports Betting: Football Player Props \- Quantopian, accessed on April 3, 2026, [https://community.quantopian.com/c/quantopian-newsletter-archives/modeling-for-sports-betting-football-player-props](https://community.quantopian.com/c/quantopian-newsletter-archives/modeling-for-sports-betting-football-player-props)  
44. Machine Learning Meets the NFL: Building a Predictive Player Performance App with Python | by Anthony Sandoval | Medium, accessed on April 3, 2026, [https://medium.com/@AnthonySandoval17/machine-learning-meets-the-nfl-building-a-predictive-player-performance-app-with-python-ac79dab465d2](https://medium.com/@AnthonySandoval17/machine-learning-meets-the-nfl-building-a-predictive-player-performance-app-with-python-ac79dab465d2)  
45. How to Spot Value Bets Using Advanced Statistical Models \- The Emory Wheel, accessed on April 3, 2026, [https://emorywheel.com/article/how-to-spot-value-bets-using-advanced-statistical-models-20260222](https://emorywheel.com/article/how-to-spot-value-bets-using-advanced-statistical-models-20260222)  
46. Programmatically Identifying Positive Expected Value (+EV) Bets | by Zachary Garrett, accessed on April 3, 2026, [https://medium.com/@zacharyrgarrett/programmatically-identifying-positive-expected-value-ev-bets-f7cfa92cd8d3](https://medium.com/@zacharyrgarrett/programmatically-identifying-positive-expected-value-ev-bets-f7cfa92cd8d3)  
47. Prices & Orderbook \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/concepts/prices-orderbook](https://docs.polymarket.com/concepts/prices-orderbook)  
48. The Polymarket API: Architecture, Endpoints, and Use Cases | by Jung-Hua Liu | Medium, accessed on April 3, 2026, [https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf](https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf)  
49. Introduction \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/api-reference/introduction](https://docs.polymarket.com/api-reference/introduction)  
50. Markets & Events \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/concepts/markets-events](https://docs.polymarket.com/concepts/markets-events)  
51. Overview \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/market-data/overview](https://docs.polymarket.com/market-data/overview)  
52. Fetching Markets \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/market-data/fetching-markets](https://docs.polymarket.com/market-data/fetching-markets)  
53. Orderbook \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/trading/orderbook](https://docs.polymarket.com/trading/orderbook)  
54. Polymarket Documentation: Overview, accessed on April 3, 2026, [https://docs.polymarket.com/](https://docs.polymarket.com/)  
55. Clients & SDKs \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/api-reference/clients-sdks](https://docs.polymarket.com/api-reference/clients-sdks)  
56. Quickstart \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/trading/quickstart](https://docs.polymarket.com/trading/quickstart)  
57. Builder Program \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/builders/overview](https://docs.polymarket.com/builders/overview)  
58. Quickstart \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/quickstart](https://docs.polymarket.com/quickstart)  
59. Overview \- Polymarket Documentation, accessed on April 3, 2026, [https://docs.polymarket.com/trading/overview](https://docs.polymarket.com/trading/overview)  
60. Python client for the Polymarket CLOB \- GitHub, accessed on April 3, 2026, [https://github.com/Polymarket/py-clob-client](https://github.com/Polymarket/py-clob-client)  
61. Polymarket | The World's Largest Prediction Market™, accessed on April 3, 2026, [https://polymarket.com/](https://polymarket.com/)  
62. 2026 FIFA World Cup Winner Predictions & Odds | Polymarket, accessed on April 3, 2026, [https://polymarket.com/event/2026-fifa-world-cup-winner-595](https://polymarket.com/event/2026-fifa-world-cup-winner-595)  
63. FIFA World Cup Props Trading Odds & Predictions | Polymarket, accessed on April 3, 2026, [https://polymarket.com/sports/fifa-world-cup/props](https://polymarket.com/sports/fifa-world-cup/props)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABg0lEQVR4Xu2VvyuFURjHHxJJDAyU/0FSQslisxEWycZiMEgxmsxWk0mIWAx2GUgpf4BCUQaSKD+/T+ec23O/99zXiwzqfOrbvedznvPep/Oe970iicTfUYfsIx/IEVJRPJ3JEvKCvCKrNMcMIOssY7SKa6bWj5v8uLJQUZ57pMN/D+s0ljnkwcxtFE/HeZTS7o+RZ3JMF3KD1BvXKe6HT4yz5G5KC0fJLXifhd42rTkjH9utQK6m+sQV9pKf8L6RPLOJNJD7dVMz4grDuQiMeK+36Dv0iFunzcbI1dSiuMI28oPej5H/Cl3zztKQ1XCBSXGF7eSHve8nn8UW8saS0GtqXSbhTOm2W8a919dFHqaQO5YR9JrbLJkacYU/efoC3cg5uXJr1e+wjKGFy+T2vLfo4W8m14KcklN4bUD9LssYsV3R8ZAZ69+OOltXZRzn0NQFqsXNHfBEOdbEHVL91IX6qmB022fNeEVKmwmZN3XTyC1yhVwgl8g18mRqEolE4t/wCTxHb6rVXAbDAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABw0lEQVR4Xu2UOyiGYRTHD7kkJVEuGawWkoEUSgaDyXWRbFYGlzIYZFIKm0VYJAuLwfSVSQyKEoOJzaKIksv5f+c8b897PJ8vyqDeX52+7/zOOU/v5XleooSEv6OY45Djg+OEIyde/pZpjneS2Y14KcY+Sc8lR56pfaGGpLlI83LNc6OOzFxxtHv5E8msT6m6Os2rNS+IOgJgoR3jTjlejAuBxf2LGNd823O2B1xwvBkXAwNDxs2qzwZ6cFOOGXVLnkN+7eUA9Yzrd5AU24wfVV9mfDYeSObcnqzV/CzqEObUu1caY4Kk2GT8oPoW479jkWSm3vjQk9pU3298mnmSYoPxveqHjQ9RwbHMccRxS3JQfLCO3T/utE4Zn2aMpNho/ID6LuOzsUAyh23hcKe7SnNslZQ6/+RGuD3VavyIeiz4E7CXMIfwyefY5bjj6OFYJ+kJfq8KSYq/OX24EfQ0Gx+6KMsxZelBcdW4A/U+2PyVXn5O0pPyHLAXdWNygHzFuBihp4K8z8tDrwUn9tXLwRZJT7fnntU51kyeEXyBcULwiwF8Kix7HJPGuU/HI8k8/nfGOohK1N+T9ODJJSQkJPxbPgHjt32Uv7+xawAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAAiklEQVR4XmNgGPTAAl2AEEgF4v9AvB1dAh94yQDRRDQwYYBoCEWXwAdAGn6hC+IDRxhIdJYQA0RDHboEPgDSQLQt14H4LgNEAwuaHAZ4AMQrgZiZAaJhGYosGgDFwS0kPl5nfQDi72hihQwQDVJo4gyfoRLYAEj8ErKADFQQ5GZsYA8DbsNGwVABAIAIIMPwtEAKAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAAjElEQVR4XmNgGAWDEUwE4lQkfgcQ1yDx4UAciC9B2blA/AuI/0P5Z4G4B8qGA5gkCPBA+fpAbAFlRyDJg4EREruMAdUADiQ2VvCJAVUDQQBSvBhdEBkIMEAUKTMg3K+FJH8ViQ0GMxkgijiB+ByUrQiVA3l8BZQNB4wMEEUg7MoAsQnGr0NSNwqGMgAAUAQcMsPKGnkAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA0CAYAAAA312SWAAADLElEQVR4Xu3dP6hcRRQH4AmChQqGiBgsghaCTQoTRAmWahUVQVTUQiyCaVJYqI0GLMRCC0mhWCWI2ogKNiKIqIVIglqoEQTF2GojioJ/57CzZHLeXe/bty9v1+T74DAzv/vm3lce7u69WwoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAW+OlHAAAsFrezwEAAKtnqGn7rdbHOWzubOPOM1IAAM6af3LQRL4rZU+08XCZvQ8AgDlsH6heNF1Pp2zq3jLclD2cAwAA1u/SWm/WujAfGNA3Yxd1815u2GI9zX7uDwAAMC4aqQvaPL6DNo9bctDEOe/r1te18fouAwBgHV6u9WWbHyizP+ac17dl7V02AAA2IJqqK3K4oL/aqGEDANgEuam6OK3nER+r9uf7sdYL3RoAYOluysH/xO9l0mi9lw/MKc6T5YYQAGDTfVLrs1qf1zpR68Nad53xFxPf52ADorm5MYdF0wMAMCo3TM+n7Opae7v1IvK1pp7LAQAAp+Um6p6U5eOLiHNNX4HR28xrAACcU+4ua5ulWB9P6yzuwk1/num1Lt/WzYfcXIbP93UOAACYiObpqW79Ta2j3Tp8kdbxfbfwZ63Xy+RpyVdbFuf7tM1nGWrYDta6LYfNdyN1+ek/BQA49ww1T734/tqxlO1p49jeIY+Uyb5DKb+h1jMpW1RcZ9kFALCwsabiklpv5bAZ25s9W+uOWjvK2r231noyZQAA570Hy9rGacgPaR17opEb2ntlDppXar3RrfPex2vtSxkAwHntozJpmv4u4y+Uzc3VyVq/tjwePPigOzb0UWA0Yz+l7Ktab3fr6c89AQCwAbkBG/NLDtZh3mtstfj//sghAMAqeTEH/2F7DkbsrnVVDldMNGzv5hAAYJXMeuVGtjMH6zB9Jcgqi1eYPJpDAAC2Vvy4/WVl8iLgd1q2v43xapNZD1QAALBFjrTxsTK56xcf1U7d380BAFiyoQcg4s4bAABLdnsbhxo2AABWQDRq17QxPJSOnerWAAAswbXd/IFuDgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA2fIvTN+oWKb7f6sAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAaCAYAAADxNd/XAAABn0lEQVR4Xu2WPShGYRTHTyEhiUGRYmCQQcpgoGxMUrJaJIuUSRallNFqkpBSlFI+yiBhsJB8lFIMksFHUWTA/3TOzXmP991ueZ+6v/rV+bi3+5z7PO8HUUJCrLT4QmgMwG+46Rsh8UAyRLA0kwzQ6xshwQN8+mJI7FPgx6iMZIBx3wgJHiDYXbiE1yQD5LreqNZ7XD1ruIHLMIdkoUspXSFrd4Z/A65MnukYpav9Oy/w3dVGSBZbaWrd8Ixk2EM4bXpMG7yDR3BWa/PwHpZqbxvOwTq4BT9giV4bsUdy3bOrp+WVMr9Vrp+a/BgemNze1w93Tc69YjgMV+Ct69VrPAZnXC9f4w5YaHp/qCK5gc98OnYodZE2boRPJvcvgfMGE+dpXKB5BO9Kq8ZD2uPn8onojC6KC/vgC9ilcbnrMZkGnyA5VhG2twEXTB4r/DbOTR49+MTlzBqc0rgWvpken/lqjXmH+AgXwRo4CNe1xzRpPRZW6XerGf5sPJq8j2QxXGs39UlK/XP4ZWKGj+GiyXdJviD4A1xh6gkJCQlZxg8/92WUYtjc+wAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAaCAYAAAD2dwHCAAAC70lEQVR4Xu2Y2etNURTHlzlTKRIPXPKAIjNF8YAMJSkZXgwlHmR4kPAvSIqkeDJHyhTJUJI8KFEoeXHDA28yZMi0vu293X2/9+x9zzn9fvceOZ/6dvf6rr3X2XffM+xzRUpK/ncmsNFihqq6svkvMFl1jc028JuNotND9YPNNtFT9YvNIvNTNYXNNnJftY/NItJXinepdJPizSmRp6rrbBYALN48NmP0Vs1VLVQttkI7U5GMNJvkJNVp1XAbz1SdUQ342yMdg1RnVWM4oYxgQ3muesBmiJtivkhI82tdO5TY5fFCNd620e+4aruYLxsbxwxR3RCzDeFxbxI8sEeS/Qa2qE548SUxB+xsYveWbapNXux+RG6nAQ8ksFYaxyE+Rx5YJY19E5lFcWjQWAnn8jBSwvV2UIx+J8lLCxYNfJf6bYg7E5M25zMkPLcg3SU+6JNqM5s5qUj8WA43p1GcyAhq+HPHDxQ6/jQJ54LsV71n02OB5CgaIU2t3ZKuX4xl0ljjQ4LnWCnhXBAMwALGSFP0nmq0F+OSSSJU65nUct+8toO3NzvFnKEhDktjDcSXyXPsksb+TcGAcWwSb1UH2ST8A+P+gUd/Eug3h00x/kMx2ye0/XvVR68Ntorpw77Pcqmfk1vMiZ7n80T1mM0YFWm+2sdUi6R5Pz+PbcIaL/apqq6wqRwSU6NqY2xbEH8Ws6DMS2n+foyzFTWgC/YzBHK4RXUYF1UrbBvF+3s5n/Wqu14cm+R0ieezEKszlWL0fUWeo4vEa2XmtphT3XFedcu2K6ovXg6TOmLbeBNwE1ltPxlckkk7/yz0Uh1l07JXzBwG23ipjbFISeCF4QCbecE/Hrj/+OBvJLcoePV55+XgvxYzCSxYVfXIyzPDVF/ZzIjbCCeBH9LdGmaLmd/AWroO7P1itTqFdV47zym/RHWKzQz0Y4PYoLqq2sgJouULB+7YTzwYcInnIfYHQSvAJhyXf9vAo78PmyUlJSUF4Q8nzK7HrBxBOgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAACK0lEQVR4Xu2WMUgdQRCGx4iYiIiFxgS7gBYSCNFKlNhY2dgpVhFsBMFOxMIiwWAhiIWlnZgUEpLGRi1EEYJaKMRYWImNiKSIoKIJOr+745sddt+z1HAf/Nzuv7Nze7e3u0eU8XDoscZjoYN1zeq2DQ+ZJ+QGPemvn/y1Wgd5Rll/WGesPtOW4i1rmdVI7l6vWFOs7zrIM0fu3uesMdMW5S+ryZfRUdBl8Iu1pOo/WeuqnqKdXC6t4yDCAb/Kl9/4+r9cc5zYgE+MX2HqArxKaxraWAusGdZHVnnYfMswa5vczAhD5PLji0iCgFpVBt/IzYiAxKnBY1D5aGV9sKbhB7lc08aXmUoiAf3+GiOVJOVrWqjw4MtYXyl886V0j/wvKBcEYbHUBRHpJClf00y5hfiF3GJfDSLijJDrM2AbLMWsHQofAt+5kBpkytdglzkwHvqsGc+CmN/WzAc66JnQfmyQKb8QR5S/3z65TaMgeCu9viwJX6uy+LGbpfxCrJDrV2N8gE9s15opkORQlbUvnJq6AG/PmobYA25576nxB1mLxrN9A65UWQJxCupOXaYuwJMDDmAwdoEhZt54l97XvGPNGg/knQUc1Z99GQlL/LX+LiLXhu1UmPCeRt5yg/JwRmC7FF6Si3mvPJwz0tdqXMVFwd8kDiUEYyaeh823PCPXvkFuZ7pgFQURRJ2sTeMB/Fqgr7xx/DJosEDtoEXYau8FgjMyMjIy/g9uAKr8rxSdd0RiAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAAB4klEQVR4Xu2UTyhnURTHj5E02PgzDQkLJUqTZjORLFhMNgrDBtlZShKNLDSzVYqajVA2koVZWdhaWLCyIGsT2VgoJWXG+b5zLued937U1Cym3qdOv3c+59z7e+/ddy9RRsa/o5hjj+MPxyFHXrz8Km0cNyTjF10tsE1SP+EocrUE1STNbzUv1/zNU8fLHHBcmfyU46vJC0jmq9U8X/Oqp44Ubjm2nDviuHMuje8kfxCo0HzXuH2OXyYHCxQflwDFQedm1b8Gelada3Q5en44h+XOOX8HSbHd+VH1Zc5bPpP09GveY2qBsFRzztep73M+YoKk+NH5AfWfnLdgydEzzDHPUcpxznFpelq0Z9I48E79jPMR30iKH5zvVT/kvOWMpOfaebhlve7UfPy5HIEHgF9xPmKMpIgnsnxR3+W8BUcHejCHBQ4BGvQaK2IJGwIbJUH4plqdH1GP4yIX6yQ99c7bm8J5h2t7RIAa9akrUUhS/Jvd103S0+S8vamQ59p9Oc8qFJecwznjbwof/3vn0IMP3bvfJsf1scnBNCXnj5H2VpDb7RqWwfdtcDyYvISkp9I4HDd+XNqLSLBJMjl+McB/mOAnx5SXzA7JmAv9bY6XI8KbwTFyz7EWL2dkZGT8PzwCrAZ8TQ6fK3sAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAAB+klEQVR4Xu2WvytGURjHH79KklKIhCIGGZRNJpMYlEWyyKSURJGySRkkGSzKJJHBH2A0mhgkWeTXIgNFIZyv85zXcx7n8N6Nup96us/zPd/33Ofee95zL1HK32FAC/+FbhPvJvr1wF8ml2zTS3yc52O5NAnyTTxrMQt2yM57bKJIjTmGTdyYeDWxoMaCwNjGOSZ3yBycsOYiWwrI+mu5zuO6KuOw4OKuRb1h4k7UQUIN3ypdck/xsRD7Ji6Vtkjf50CNp6q1LqV5wFAtcrBL9omESNo8vKtKa2fdsaxqB7QzLUpgQIzw8TeSNO+WyKzS61jv4xr/odCcrrcolfRlQjyZaPQcPkmabyXrnVA6NgPo01zHmozpHrhDh+RfRInn+CJJ851kvWNKL2V9jetYkzE9CIzySYRI0nwTWe+40stYn+M6dr6YnuHcxBDnztgick2S5nPIemeUXsP6INfYHEJz/to8Bi9ELvUQSZoH8MZ2G7fX73GtgfamRcmLyN0E9SLX/NR8oYlRpeHkR0qbIn8Ot1Q10Ca1KMEeu8k5zO6NiPUa4pHCJwLQEc1C62BNgnpFabjILVH30PffBcHXpFt3eBIV/vAnD2S/O7DEEFdkX98NwtNr4kDUDnent8nu6ev+cAaMnZJ9K8Nf7A//TFZXmpKSkpLyL/gANECmmB9qUVkAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAnCAYAAACylRSjAAAD+UlEQVR4Xu3cy8t1UxgA8OV+yzWFGHxyT8klMhCRopigFDHAN2AiExNhopDLwMTALZLLH+ASKUVJSGYMiBhSRMjdejp7d9Z53r3P4X339/Vxfr96ep/nWevsyzvZq33O3qUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/JV/lxpq4OjcAgPW2e42/RuKhGrs19W/dZ3p9/73Un8Ifqb6wzPb1XOpnB5bZZz9M/YfL7PPnpf5WHVDj+xrv54ERcQytm7re46n/baqncFeZ7Sv2ucwXNX6ucVDq31hmn78/9cfOAQCY0B5l40Iiemc0dR4PP+bGRA6vcVFTb6vxYlMf0uSt62rc0+VxvGd2+ac19unyB2uc2+VTaBeWQ/+j1iNlcc61TR6+TnUslqfyZI1Tu/yyGrc3Y613mzyO9eQuv77GzV1+RPc3rDoHAGAiL5XFhcelTd7Li5H9y8Y7MFPJ+1pV99p+HF8vz89179DcKIuLk+yjsvj15dh2e2eXxTmRt4uy/PlfUr0Vedu57rX9WHz1dZ4fi+Cw6hwAgInERfbyLj+hHWjEnIObOr4y21HyRX9VHeLYot/fxbqhGcvzc92Lr12PauoTuxgT24lFWFuP6RfE7ZwLuvqJMls0792MhWXb+7fytnI9JOZ81uStvl51DgDAROKCe2+NR7t8SPRv6fKnm/6Y02o8m+KZGk+V2cX9+PnUDfIxrKrDNWWx/2eNY7o8z8916+IaR9c4qcYpaSyL7cR5tvWQO5s8z4nf5EUv98NQb7PytnI9pJ2T57f1snMAACawZxm+MH/c9EL03+ry7e3ADpAv/KvqkL9ufKWp8/xcZ7eW+R3HZWI7Z6V6yCVN3s75ssm/Kxs/n+te9JfFkNzPdZbHx+pV5wAATOD1Gr/nZnV3qt8oyxcE2bE1HlgS/Q/gh+R9rKp7bf/lps7zc92KBxXih/bnlPkdujE/lNW/YYvxt5uIOfE35Pmr6q3I28p1qx0b+x/GHcyQ+7kGACYQF9j2qctYaA1ddPvXauwMeT+H1fi8y4+ssVeX31YW5/7U5G3/tTJfIMYrSLbNhxZ8kxvVJ7mRDC1u+lehDGn78TqQfZs6FoCtsW1sxh1l/pRnPCl7ZZefXjaeQxv3df14HUr/pG48ZdpbdQ4AwE429kDC1IYWKvHU5ztl8SvIIfH6ibgbmMUi480a++WBCbxa46rc/Ifi69IPalyRB6rHcmOLYnEWr+2I16ZsRjyQEQve41J/2TkAAP9jL+TGmokX8gIA7NKez40182tuAADsivrfXK2b83MDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANbP3wYY/Jm4LxxTAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAnCAYAAACylRSjAAAD5ElEQVR4Xu3cS6h9UxgA8OURIqWIhJTHxICBZIAkAwOvvMujRMlMZCQGFEkm5Jn3q8hASpREmXgOkKRMpJQ3kWde6+vs1fnOau/r+t99kvr96uus71v7nL3uuoO92nudUwoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwP/UXxPx9tD/x5D/OeRNO+6Lrr5Vu9X4vsZbfUfn/Rq/1zioq19aFuO6r6vvWBZ/w+ddfQ731/ikL3aOLYtxvd53VG/U+LovVj/U+LXGDn3HDA6p8VuNl/qOziVlMe6bu/rUPAMAaxIX3t6VqT3W/25fmEksEJux84ZvUjuOuXBon5/q4cvUvi61pz53W2xmvKek9gll9bipdl4If5jac9iuxscpj4XvmItqXD6096nx3NDeaJ4BgDW4uowvNPZM7bH+o/vCDN6pcW7Kx84bcv3plMdrLEaaVt+jxpOpnhd8W5XHEnfDDkh5EwuisYXZ3l39xbJcfP6S6lek9hzizlqcu9nMPOd8ap4BgDWJi+2tQ3uX3JHEMad3+TrE5x7V5YemfEw8NmyPa48vi/fEI8q4G7RTO2ioR+xa49FU36o8F3En7ImUT2nvebjGj6l+T41Ph3Ybb/hqeJ1L///r86avt3yjeQYA1iAuvDfVuGNoj4n6U0P74lSfsnuNx7p4pMaDZXGRP3l56Io4z+FdfkHKx/RjjgVTXuw0sRht9bO7vm21fVk9Tzwm/ijlY94ry/15r5bVu323ldXPa+N9NtWyqXl+qMYDZXp/WT83ke/X1cLYcc3UPAMAa5AvuD8Pr/emWvigLPdqPZ47ZhZjObLLj0h5r18s5I3/35VlfyzW4u5VaJvle23xNRVTcl/M0zMp7x1Y45WU314Wdwibu2p8O7Tb/2L/sjjHmUM+h/7v6fOmr7d8ap4BgDW4poxfbF/u8mvL4rifuvqUnWvcskGcujx0RSxeNrOHLeSxtOP646fqx3X5VuTPjj1s56U8iwXhjV3tmLL6/tjDdneNO1OtaQu47J/mOWJMnHOre9iyPgcAZhQX2tNSvtdQ68VPYozV1yGfZ6N2jteGevwcSN6H1+5eHVwWi6Em7xvbqs/K8pz9GNtPiJw45DmaqXb+5ubzqT2X9vnx5YGYtybGcNjQjp8iaV/WiMfYcUcwTM0zAPAfO6cvrNELNc7qi5t0Uln8htwZXT0WnfG513f1OVxW46q++C/E3r4b+mJZfAO2PcqdWyzU4k5qLCY3sm+NN8ti0ZtNzTMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADDlb7LSCKFSHSwOAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAZCAYAAACRiGY9AAAB9ElEQVR4Xu2WMUhWURTH/yKBgxq56FQQiIRT0CIIEoJjtLW02yCIQlOCS4OLS5ht4tLipAhttQaCg5JQiFqKUVCiCYqF2vlz3+s733nn691PDBruD/74zu9c77vX7333CSQSif+Ru5L3knPJS9MrY1LyU/Jd0mt6OV2SQ4T5lyWN1e3LZ1RypupHCDeP4UDyVNXHkglVk+uSt6puQZj/qnLRXJE8ttKBN7jlOLs4Sz+Km29znK3JkOSDlTE0oXxT9+Hf9AS+1/CR88bQPTT1bVWTYclH46JoRvmm3sBf2BZ8r2H/l5UIftXUzJRxraqOhs9u2ab24S9+Db7XsP/DSgTP71ZOe+bynEo6Vb8uYjaV38iyAt9r2N+zEv6cPcozC9VtnztO+iTPHM/kfEZxAeQdfK9hn5+0hV6fpmOST9n1ICobm/kzogb3nDyQzDqeyan1ndqA7zXs80Cx0K9n13wfefMcwfelxDx+T+BPHnP65X9xC92L7Pq5ZEf1NN7vlhKzKcLJ+X6xbtG4cVNPo7iwhszxHUlGstqjlv8rsZv6gnCE53SgemFkPnNzyhG6blUvoXgicsyAca9Q+TTrInZT5JvkK8LNuIib1W1ck+yi+G65gTD+tWQbYYyF/wTwGOe4zewnH/sLUc+mEolEIpH4F/wGfVWOeT2OVrgAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAAB5klEQVR4Xu2WyytFURTGl2eUZELKTDGQmX9CDMzITEwUM2VmqJSSP8BQ8shrbCgTJpQomchzIAMKIazP3vvedZa9zr3De+v86uus9a3Vuvvcc+7elyijdBjWRrnQx/phDelCKVNJbtEL/jrrr82yyTPDema9ssZUrRjwBa1pUzDKumd9seZULQoae3yMRQdkDM5YeyI/ZR2I3GKa9UJuHrSeLOfYYN2JfJn1JPIosQU/Kr9R5QF4TdpMIW3xqFVHvF7lJUBDm4jBNrknEjgWNQm8JW2mYC1+kez5l9qUoAEa99cYoUdj+RbW4j8oPqfg/FbKN0FvrI5Ehz3E8i2sxVtzLD9BFeuEkjeB9zxgDbF8C/Tih6mx5lh+FDTKJyH92BDLt0DvpjbJnmP5Oa5YIz4Ojd0iDn5siOVboHdLm+Q2h9icgvNRvBax9ANhn9bAO9dmCujf0Sa588Oa/61NyaeIw4B2EYNBlQfghQMO1LEmRK5B/642Kf+qauBNaVOCPXbFx2iu8dfOXEe+hu00MO89CXKoS/mgllzNOpXxDa+KvJ/+z4+Cf5PhvcOTaEmW/6gnVz8ktzO9syoSHUQDrCPlTZI7sW/JvZ43rAdyW7IG+/0Fa5/cZzUky+kUdacZGRkZGWXBLw0fqbSISLhKAAAAAElFTkSuQmCC>